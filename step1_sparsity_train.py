"""
step1_sparsity_train.py
───────────────────────
Phase 1 of BN-gamma pruning pipeline for UltraFace (mb_tiny_fd).

What this does:
  Continues training your already-trained model for a short run (10-15 epochs)
  while adding an L1 penalty on every BN gamma (scale) parameter in the
  PRUNABLE blocks. This pushes unimportant channels toward gamma ≈ 0 without
  changing the model architecture at all.

  Prunable blocks (backbone only, non-tap):
      base_net[0..7], base_net[9], base_net[10]

  Frozen / excluded from L1 penalty (FPN tap outputs — heads depend on them):
      base_net[8]  → 128ch  → cls/reg header[0]
      base_net[11] → 256ch  → cls/reg header[1]
      base_net[12] → 256ch  → extras + cls/reg header[2]
      All extras, classification_headers, regression_headers BNs

Usage:
    python step1_sparsity_train.py \
        --weights models/your_trained.pth \
        --epochs 15 \
        --sparsity_lambda 1e-4 \
        --lr 0.001 \
        --batch_size 24

Output:
    models/pruning/sparsity_trained.pth   ← load this into step2
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project imports (same as train.py) ───────────────────────────────────────
from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser()
parser.add_argument('--weights',          required=True,  help='Path to trained .pth (state dict)')
parser.add_argument('--epochs',           default=15,     type=int)
parser.add_argument('--sparsity_lambda',  default=1e-4,   type=float,
                    help='L1 penalty weight on BN gammas. Higher = more zeros pushed.')
parser.add_argument('--lr',               default=1e-3,   type=float)
parser.add_argument('--momentum',         default=0.9,    type=float)
parser.add_argument('--weight_decay',     default=5e-4,   type=float)
parser.add_argument('--batch_size',       default=64,     type=int)
parser.add_argument('--num_workers',      default=6,      type=int)
parser.add_argument('--input_size',       default=320,    type=int)
parser.add_argument('--overlap_threshold',default=0.35,   type=float)
parser.add_argument('--output_dir',       default='models/pruning')
parser.add_argument('--debug_steps',      default=100,    type=int)
args = parser.parse_args()

define_img_size(args.input_size)

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing1 import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior
from vision.nn.multibox_loss import MultiboxLoss
from vision.datasets.hagrid_dataset import HaGRIDDataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_ROOT = os.path.join('data', 'wider_face_add_lm_10_10', 'dataset')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prunable BN indices in base_net (non-tap blocks only)
# Each conv_dw block has TWO BN layers: index .1 (depthwise BN) and .4 (pointwise BN)
# conv_bn block (index 0) has ONE BN: index .1
PRUNABLE_BASE_NET_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


def get_prunable_bn_layers(net):
    """Return list of (name, BN module) for all prunable BN gammas."""
    bns = []
    base_net = net.base_net  # nn.Sequential of 13 blocks

    for idx in PRUNABLE_BASE_NET_INDICES:
        block = base_net[idx]
        for name, module in block.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                full_name = f'base_net[{idx}].{name}'
                bns.append((full_name, module))
    return bns


def sparsity_loss(prunable_bns, lam):
    """L1 penalty on BN gamma (weight) parameters."""
    penalty = torch.tensor(0.0, device=DEVICE)
    for _, bn in prunable_bns:
        penalty += bn.weight.abs().sum()
    return lam * penalty


def train_epoch(loader, net, criterion, optimizer, prunable_bns, epoch):
    net.train()
    total_loss = 0.0
    for i, (images, boxes, labels) in enumerate(loader):
        images = images.to(DEVICE)
        boxes  = boxes.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        confidence, locations = net(images)
        reg_loss, clf_loss = criterion(confidence, locations, labels, boxes)
        task_loss = reg_loss + clf_loss

        # Add sparsity penalty on prunable BN gammas
        s_loss = sparsity_loss(prunable_bns, args.sparsity_lambda)
        loss = task_loss + s_loss
        loss.backward()
        optimizer.step()

        total_loss += task_loss.item()
        if i and i % args.debug_steps == 0:
            logging.info(f'  Epoch {epoch} step {i}: '
                         f'task={task_loss.item():.4f}  sparsity={s_loss.item():.6f}')

    return total_loss / len(loader)


def val_epoch(loader, net, criterion):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels in loader:
            images = images.to(DEVICE)
            boxes  = boxes.to(DEVICE)
            labels = labels.to(DEVICE)
            confidence, locations = net(images)
            r, c = criterion(confidence, locations, labels, boxes)
            total_loss += (r + c).item()
    return total_loss / len(loader)


def print_gamma_stats(prunable_bns):
    """Show min/mean/max gamma per prunable block — useful to pick threshold in step2."""
    logging.info('\n── BN Gamma statistics after sparsity training ──')
    all_gammas = []
    for name, bn in prunable_bns:
        g = bn.weight.abs().detach().cpu()
        all_gammas.append(g)
        logging.info(f'  {name:<35} min={g.min():.4f}  mean={g.mean():.4f}  '
                     f'max={g.max():.4f}  near-zero(<0.01): {(g < 0.01).sum().item()}/{len(g)}')
    flat = torch.cat(all_gammas)
    logging.info(f'\n  GLOBAL  min={flat.min():.4f}  mean={flat.mean():.4f}  '
                 f'p10={flat.kthvalue(max(1,int(0.10*len(flat))))[0]:.4f}  '
                 f'p20={flat.kthvalue(max(1,int(0.20*len(flat))))[0]:.4f}  '
                 f'near-zero(<0.01): {(flat < 0.01).sum().item()}/{len(flat)}')


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model and load weights ─────────────────────────────────────────
    logging.info('Building model ...')
    net = create_mb_tiny_fd(num_classes=3, is_test=False, device=str(DEVICE))
    state = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state)
    net.to(DEVICE)
    logging.info(f'Loaded weights from {args.weights}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_transform  = TrainAugmentation(fd_config.image_size, fd_config.image_mean, fd_config.image_std)
    test_transform   = TestTransform(fd_config.image_size, fd_config.image_mean_test, fd_config.image_std)
    target_transform = MatchPrior(fd_config.priors, fd_config.center_variance,
                                  fd_config.size_variance, args.overlap_threshold)

    train_dataset = HaGRIDDataset(root=DATASET_ROOT, transform=train_transform,
                                  target_transform=target_transform, is_test=False)
    val_dataset   = HaGRIDDataset(root=DATASET_ROOT, transform=test_transform,
                                  target_transform=target_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    logging.info(f'Train: {len(train_dataset):,}  Val: {len(val_dataset):,}')

    # ── Identify prunable BN layers ───────────────────────────────────────────
    prunable_bns = get_prunable_bn_layers(net)
    logging.info(f'Prunable BN layers found: {len(prunable_bns)}')
    for name, bn in prunable_bns:
        logging.info(f'  {name}  ({bn.num_features} channels)')

    # ── Optimizer (lower LR — we're fine-tuning, not training from scratch) ───
    criterion = MultiboxLoss(fd_config.priors, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, net, criterion, optimizer, prunable_bns, epoch)
        val_loss   = val_epoch(val_loader, net, criterion)
        scheduler.step()

        logging.info(f'Epoch {epoch:3d}/{args.epochs}  '
                     f'train={train_loss:.4f}  val={val_loss:.4f}  '
                     f'lr={optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(args.output_dir, 'sparsity_trained.pth')
            net.save(ckpt)
            logging.info(f'  ✓ Saved best → {ckpt}')

    # ── Print gamma stats to help choose threshold in step2 ──────────────────
    # Reload best checkpoint for accurate stats
    net.load_state_dict(torch.load(os.path.join(args.output_dir, 'sparsity_trained.pth'),
                                   map_location='cpu'))
    net.to(DEVICE)
    prunable_bns = get_prunable_bn_layers(net)
    print_gamma_stats(prunable_bns)

    logging.info('\nDone. Next: run step2_analyze_and_prune.py')
