"""
step3_finetune.py
─────────────────
Phase 3 of BN-gamma pruning pipeline for UltraFace (mb_tiny_fd).

CORRECTED VERSION — fixes incorrect tap indices that caused size mismatch errors.

The SSD forward pass uses source_layer_indexes = [8, 11, 13], which means:
    base_net[0:8]  → tap 1 (output of block 7,  64ch)  → header 0
    base_net[8:11] → tap 2 (output of block 10, 128ch) → header 1
    base_net[11:13]→ tap 3 (output of block 12, 256ch) → header 2
    extras output  → tap 4 (256ch)                     → header 3

So FIXED blocks (cannot prune output) are 7, 10, 12 — not 8, 11, 12.

What this does:
  Fine-tunes the pruned model to recover accuracy lost during pruning.
  Uses the same loss, optimizer, and scheduler as your original train.py.
  Saves the best checkpoint by val loss.

Usage:
    python step3_finetune.py \
        --weights models/pruning/pruned_model.pth \
        --pruned_widths "12,24,24,24,52,56,60,64,112,88,128,248,256" \
        --epochs 20 \
        --lr 0.01

    The --pruned_widths string must match exactly what step2 produced.
    Copy the "Pruned" column from step2's channel width comparison output.
    Fixed tap outputs (indices 7, 10, 12) are always 64, 128, 256.

python step3_finetune.py --weights models/pruning/pruned_model.pth --pruned_widths "12,24,20,24,44,52,56,64,96,56,128,100,256" --epochs 20 --lr 0.01
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser()
parser.add_argument('--weights',       required=True,  help='Pruned model .pth from step2')
parser.add_argument('--pruned_widths', required=True,
                    help='Comma-separated list of 13 channel widths from step2 output. '
                         'E.g. "12,24,24,24,52,56,60,64,112,88,128,248,256"')
parser.add_argument('--epochs',        default=20,    type=int)
parser.add_argument('--lr',            default=1e-2,  type=float)
parser.add_argument('--momentum',      default=0.9,   type=float)
parser.add_argument('--weight_decay',  default=5e-4,  type=float)
parser.add_argument('--batch_size',    default=64,    type=int)
parser.add_argument('--num_workers',   default=6,     type=int)
parser.add_argument('--input_size',    default=320,   type=int)
parser.add_argument('--overlap_threshold', default=0.35, type=float)
parser.add_argument('--scheduler',     default='cosine', choices=['cosine', 'multi-step'])
parser.add_argument('--milestones',    default='10,16', type=str)
parser.add_argument('--validation_epochs', default=1, type=int)
parser.add_argument('--debug_steps',   default=100,   type=int)
parser.add_argument('--output_dir',    default='models/pruning')
args = parser.parse_args()

define_img_size(args.input_size)

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing1 import TrainAugmentation, TestTransform
from vision.ssd.ssd import SSD, MatchPrior
from vision.nn.multibox_loss import MultiboxLoss
from vision.datasets.hagrid_dataset import HaGRIDDataset
from vision.ssd.mb_tiny_fd import SeperableConv2d

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_ROOT = os.path.join('data', 'wider_face_add_lm_10_10', 'dataset')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_pruned_backbone(widths):
    """Rebuild the pruned backbone from the width list."""
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    w = widths
    return nn.Sequential(
        conv_bn(3,     w[0],  2),
        conv_dw(w[0],  w[1],  1),
        conv_dw(w[1],  w[2],  2),
        conv_dw(w[2],  w[3],  1),
        conv_dw(w[3],  w[4],  2),
        conv_dw(w[4],  w[5],  1),
        conv_dw(w[5],  w[6],  1),
        conv_dw(w[6],  w[7],  1),   # tap 1 — output FIXED at 64
        conv_dw(w[7],  w[8],  2),
        conv_dw(w[8],  w[9],  1),
        conv_dw(w[9],  w[10], 1),   # tap 2 — output FIXED at 128
        conv_dw(w[10], w[11], 2),
        conv_dw(w[11], w[12], 1),   # tap 3 — output FIXED at 256
    )


def build_pruned_ssd(widths):
    """Rebuild the full SSD with pruned backbone and matching heads."""
    from torch.nn import ModuleList, Sequential, Conv2d, ReLU
    num_classes = 3
    backbone = build_pruned_backbone(widths)

    # CORRECTED: tap channels come from blocks 7, 10, 12 (not 8, 11, 12)
    tap1_ch  = widths[7]   # always 64 (fixed)
    tap2_ch  = widths[10]  # always 128 (fixed)
    tap3_ch  = widths[12]  # always 256 (fixed)
    extra_ch = 256         # extras output is fixed at 256

    extras = nn.ModuleList([
        nn.Sequential(
            Conv2d(tap3_ch, 64, kernel_size=1),
            ReLU(),
            SeperableConv2d(64, extra_ch, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])
    regression_headers = nn.ModuleList([
        SeperableConv2d(tap1_ch, 3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(tap2_ch, 2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(tap3_ch, 2 * 4, kernel_size=3, padding=1),
        Conv2d(extra_ch, 3 * 4, kernel_size=3, padding=1),
    ])
    classification_headers = nn.ModuleList([
        SeperableConv2d(tap1_ch, 3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(tap2_ch, 2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(tap3_ch, 2 * num_classes, kernel_size=3, padding=1),
        Conv2d(extra_ch, 3 * num_classes, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, backbone, [8, 11, 13],
               extras, classification_headers, regression_headers,
               is_test=False, config=fd_config, device=str(DEVICE))


def train_epoch(loader, net, criterion, optimizer, epoch):
    net.train()
    total_loss = reg_total = clf_total = 0.0
    for i, (images, boxes, labels) in enumerate(loader):
        images = images.to(DEVICE)
        boxes  = boxes.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        conf, loc = net(images)
        reg_loss, clf_loss = criterion(conf, loc, labels, boxes)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        reg_total  += reg_loss.item()
        clf_total  += clf_loss.item()
        if i and i % args.debug_steps == 0:
            logging.info(f'  [{epoch}:{i}] loss={loss.item():.4f}  '
                         f'reg={reg_loss.item():.4f}  clf={clf_loss.item():.4f}')
    n = len(loader)
    return total_loss / n, reg_total / n, clf_total / n


def val_epoch(loader, net, criterion):
    net.eval()
    total_loss = fg_correct = fg_total = 0
    with torch.no_grad():
        for images, boxes, labels in loader:
            images = images.to(DEVICE)
            boxes  = boxes.to(DEVICE)
            labels = labels.to(DEVICE)
            conf, loc = net(images)
            r, c = criterion(conf, loc, labels, boxes)
            total_loss += (r + c).item()
            _, pred = torch.max(conf, 2)
            flat_pred   = pred.view(-1)
            flat_labels = labels.view(-1)
            fg_mask = flat_labels != 0
            fg_correct += (flat_pred[fg_mask] == flat_labels[fg_mask]).sum().item()
            fg_total   += fg_mask.sum().item()

    avg_loss = total_loss / len(loader)
    fg_acc   = fg_correct / fg_total * 100 if fg_total > 0 else 0.0
    return avg_loss, fg_acc


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse widths from step2 output ────────────────────────────────────────
    widths = [int(x.strip()) for x in args.pruned_widths.split(',')]
    assert len(widths) == 13, f'Expected 13 widths, got {len(widths)}'
    logging.info(f'Pruned channel widths: {widths}')

    # Sanity check: tap blocks must have their fixed widths
    expected_taps = {7: 64, 10: 128, 12: 256}
    for idx, expected in expected_taps.items():
        if widths[idx] != expected:
            logging.error(f'Tap block {idx} has width {widths[idx]}, expected {expected}. '
                          f'This will cause a size mismatch.')
            sys.exit(1)
    logging.info('Tap widths verified (bn[7]=64, bn[10]=128, bn[12]=256)')

    # ── Build model and load pruned weights ───────────────────────────────────
    net = build_pruned_ssd(widths)
    state = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state)
    net.to(DEVICE)

    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f'Pruned model params: {total_params:,}')

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

    # ── Criterion & Optimizer ─────────────────────────────────────────────────
    criterion = MultiboxLoss(fd_config.priors, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        milestones = [int(x) for x in args.milestones.split(',')]
        scheduler  = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    logging.info(f'\nFine-tuning for {args.epochs} epochs on {DEVICE} ...')

    for epoch in range(args.epochs):
        train_loss, reg_loss, clf_loss = train_epoch(train_loader, net, criterion, optimizer, epoch)
        scheduler.step()
        logging.info(f'Epoch {epoch:3d}/{args.epochs}  '
                     f'train={train_loss:.4f} (reg={reg_loss:.4f} clf={clf_loss:.4f})  '
                     f'lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch % args.validation_epochs == 0 or epoch == args.epochs - 1:
            val_loss, fg_acc = val_epoch(val_loader, net, criterion)
            logging.info(f'           val={val_loss:.4f}  fg_acc={fg_acc:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = os.path.join(args.output_dir, f'finetuned_best.pth')
                net.save(ckpt)
                logging.info(f'  ✓ Best saved → {ckpt}')

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, 'finetuned_final.pth')
    net.save(final_path)
    logging.info(f'\nFinal model → {final_path}')
    logging.info(f'Best val loss: {best_val_loss:.4f}')
    logging.info('\nDone. Compare finetuned_best.pth against your original model.')