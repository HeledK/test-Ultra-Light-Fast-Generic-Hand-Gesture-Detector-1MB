"""
train.py
--------
Training script for Mb-Tiny SSD gesture detector.
Uses HaGRID JSON annotations instead of VOC XML.

Dataset path (relative to this file):
    data/wider_face_add_lm_10_10/dataset/

    dataset/
    ├── palm_train.json
    ├── palm_val.json
    ├── two_train.json
    ├── two_val.json
    ├── train/
    │   ├── palm/
    │   └── two/
    └── val/
        ├── palm/
        └── two/

Example usage:
    python train.py --net slim --input_size 320 --num_epochs 200 \
        --lr 0.01 --scheduler multi-step --milestones 80,100 \
        --checkpoint_folder models/
"""

import argparse
import itertools
import logging
import os
import sys
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset
import torch.onnx

from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels

# ── Dataset ───────────────────────────────────────────────────────────────────
from vision.datasets.hagrid_dataset import HaGRIDDataset

# ── Dataset root — adjust if your folder layout differs ──────────────────────
DATASET_ROOT = os.path.join('data', 'wider_face_add_lm_10_10', 'dataset')
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Train gesture detector with Pytorch')

parser.add_argument('--net', default="slim",
                    help="Network architecture: RFB or slim")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all layers except prediction head.")

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--base_net_lr', default=None, type=float)
parser.add_argument('--extra_layers_lr', default=None, type=float)

# Pretrained / checkpoint
parser.add_argument('--base_net',       help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained SSD model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint .pth to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler: multi-step, cosine, or poly")
parser.add_argument('--milestones', default="80,100", type=str,
                    help="Epoch milestones for MultiStepLR")
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max for CosineAnnealingLR')

# Training params
parser.add_argument('--batch_size',        default=24,  type=int)
parser.add_argument('--num_epochs',        default=200, type=int)
parser.add_argument('--num_workers',       default=4,   type=int)
parser.add_argument('--validation_epochs', default=5,   type=int)
parser.add_argument('--debug_steps',       default=100, type=int)
parser.add_argument('--use_cuda',          default=True, type=str2bool)
parser.add_argument('--checkpoint_folder', default='models/')
parser.add_argument('--log_dir',           default='./models/logs')
parser.add_argument('--cuda_index',        default="0", type=str)
parser.add_argument('--power',             default=2,   type=int)
parser.add_argument('--overlap_threshold', default=0.35, type=float)
parser.add_argument('--optimizer_type',    default="SGD", type=str)
parser.add_argument('--input_size',        default=320, type=int,
                    help='Input size: 128/160/320/480/640/1280')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

# Must call before importing fd_config
define_img_size(args.input_size)

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing1 import TrainAugmentation, TestTransform  #changed to preprocessing
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
)
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA.")


# ── LR helpers ────────────────────────────────────────────────────────────────
def lr_poly(base_lr, iter_):
    return base_lr * ((1 - float(iter_) / args.num_epochs) ** args.power)


def adjust_learning_rate(optimizer, epoch):
    lr = lr_poly(args.lr, epoch)
    optimizer.param_groups[0]['lr'] = lr


# ── Train / val loops ─────────────────────────────────────────────────────────
def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = running_reg_loss = running_clf_loss = 0.0
    running_correct = running_total = 0

    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, labels = data
        images = images.to(device)
        boxes  = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        reg_loss, clf_loss    = criterion(confidence, locations, labels, boxes)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()

        running_loss     += loss.item()
        running_reg_loss += reg_loss.item()
        running_clf_loss += clf_loss.item()

        _, predicted  = torch.max(confidence, 2)
        predicted     = predicted.view(-1)
        labels_flat   = labels.view(-1)
        running_correct += (predicted == labels_flat).sum().item()
        running_total   += labels_flat.size(0)

        if i and i % debug_steps == 0:
            print(".", flush=True)
            logging.info(
                f"Epoch: {epoch}, Step: {i}, "
                f"Train Loss: {running_loss/debug_steps:.4f}, "
                f"Regression Loss: {running_reg_loss/debug_steps:.4f}, "
                f"Classification Loss: {running_clf_loss/debug_steps:.4f}"
            )
            running_loss = running_reg_loss = running_clf_loss = 0.0
            running_correct = running_total = 0

""" 
def test(loader, net, criterion, device):
    net.eval()
    running_loss = running_reg_loss = running_clf_loss = 0.0
    running_correct = running_total = 0
    num = 0

    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes  = boxes.to(device)
        labels = labels.to(device)
        num   += 1

        with torch.no_grad():
            confidence, locations = net(images)
            reg_loss, clf_loss    = criterion(confidence, locations, labels, boxes)
            loss = reg_loss + clf_loss

        running_loss     += loss.item()
        running_reg_loss += reg_loss.item()
        running_clf_loss += clf_loss.item()

        _, predicted  = torch.max(confidence, 2)
        predicted     = predicted.view(-1)
        labels_flat   = labels.view(-1)
        running_correct += (predicted == labels_flat).sum().item()
        running_total   += labels_flat.size(0)

    avg_loss     = running_loss     / num
    avg_reg_loss = running_reg_loss / num
    avg_clf_loss = running_clf_loss / num
    accuracy     = (running_correct / running_total) * 100 if running_total > 0 else 0.0

    return avg_loss, avg_reg_loss, avg_clf_loss, accuracy
"""
#added test
def test(loader, net, criterion, device): 
    net.eval()
    running_loss = running_reg_loss = running_clf_loss = 0.0
    # Separate counters for foreground-only accuracy
    fg_correct = fg_total = 0
    # Also track positive detection rate (recall-ish)
    bg_as_fg_count = 0  # false positives at anchor level
    num = 0

    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            reg_loss, clf_loss = criterion(confidence, locations, labels, boxes)
            loss = reg_loss + clf_loss

        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        running_clf_loss += clf_loss.item()

        _, predicted = torch.max(confidence, 2)
        predicted = predicted.view(-1)
        labels_flat = labels.view(-1)

        # Foreground anchor accuracy (label != 0)
        fg_mask = labels_flat != 0
        fg_correct += (predicted[fg_mask] == labels_flat[fg_mask]).sum().item()
        fg_total += fg_mask.sum().item()

        # Background anchors incorrectly predicted as foreground
        bg_mask = labels_flat == 0
        bg_as_fg_count += (predicted[bg_mask] != 0).sum().item()

    avg_loss = running_loss / num
    avg_reg = running_reg_loss / num
    avg_clf = running_clf_loss / num
    fg_acc = (fg_correct / fg_total * 100) if fg_total > 0 else 0.0
    # bg_as_fg_count is absolute; could normalize by total bg anchors for rate

    return avg_loss, avg_reg, avg_clf, fg_acc

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    timer = Timer()
    logging.info(args)

    # Select architecture
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config     = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config     = fd_config
    else:
        logging.fatal("Unknown --net value. Use 'slim' or 'RFB'.")
        sys.exit(1)

    # Transforms
    train_transform  = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    test_transform   = TestTransform(config.image_size, config.image_mean_test, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)

    os.makedirs(args.checkpoint_folder, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    logging.info("Loading training dataset …")
    train_dataset = HaGRIDDataset(
        root             = DATASET_ROOT,
        transform        = train_transform,
        target_transform = target_transform,
        is_test          = False,
    )

    logging.info("Loading validation dataset …")
    val_dataset = HaGRIDDataset(
        root             = DATASET_ROOT,
        transform        = test_transform,
        target_transform = target_transform,
        is_test          = True,
    )

    num_classes = len(train_dataset.class_names)   # 3: BACKGROUND, palm, two
    logging.info(f"Classes ({num_classes}): {train_dataset.class_names}")

    # Save label file for reference
    label_file = os.path.join(args.checkpoint_folder, "labels.txt")
    store_labels(label_file, train_dataset.class_names)
    logging.info(f"Labels saved to {label_file}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
    )

    logging.info(f"Train: {len(train_dataset):,} images | "
                 f"Val: {len(val_dataset):,} images")

    # ── Build model ───────────────────────────────────────────────────────────
    logging.info("Building network …")
    net = create_net(num_classes)

    cuda_index_list = None
    if torch.cuda.device_count() >= 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info(f"Using GPU(s): {cuda_index_list}")

    last_epoch  = -1
    base_net_lr = args.base_net_lr    if args.base_net_lr    is not None else args.lr
    extra_lr    = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    # Parameter groups
    if args.freeze_base_net:
        logging.info("Freezing base net.")
        freeze_net_layers(net.base_net)
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()), 'lr': extra_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters())}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters())
        logging.info("Frozen all layers except prediction heads.")
    else:
        mod = net.module if cuda_index_list else net
        params = [
            {'params': mod.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                mod.source_layer_add_ons.parameters(),
                mod.extras.parameters()), 'lr': extra_lr},
            {'params': itertools.chain(
                mod.regression_headers.parameters(),
                mod.classification_headers.parameters())}
        ]

    # Load weights
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        net.module.load(args.resume) if cuda_index_list else net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained SSD {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Model loaded in {timer.end("Load Model"):.2f}s')

    net.to(DEVICE)

    criterion = MultiboxLoss(
        config.priors,
        neg_pos_ratio   = 3,
        center_variance = 0.1,
        size_variance   = 0.2,
        device          = DEVICE,
    )

    # Optimizer
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
        logging.info("Using Adam optimizer.")
    else:
        logging.fatal(f"Unsupported optimizer: {args.optimizer_type}")
        sys.exit(1)

    # Scheduler
    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler  = MultiStepLR(optimizer, milestones=milestones,
                                     gamma=0.1, last_epoch=last_epoch)
            logging.info(f"MultiStepLR milestones: {milestones}")
        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, args.t_max,
                                          last_epoch=last_epoch)
            logging.info("CosineAnnealingLR scheduler.")
        elif args.scheduler == 'poly':
            logging.info("PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported scheduler: {args.scheduler}")
            sys.exit(1)

    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Starting training from epoch {last_epoch + 1} …")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(last_epoch + 1, args.num_epochs):

        if args.optimizer_type != "Adam":
            if args.scheduler != "poly" and epoch != 0:
                scheduler.step()

        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)

        logging.info(f"LR: {optimizer.param_groups[0]['lr']}")

        # Validation
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            print("\nValidating …")
            val_loss, val_reg, val_clf, val_acc = test(
                val_loader, net, criterion, DEVICE)

            logging.info(
                f"Epoch: {epoch} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Reg Loss: {val_reg:.4f} | "
                f"Clf Loss: {val_clf:.4f} | "
                f"Val Accuracy: {val_acc:.2f}%"
            )

            # Save checkpoint
            folder = os.path.join(
                args.checkpoint_folder,
                f"{args.net}-Epoch-{epoch}-Loss-{val_loss}"
            )
            os.makedirs(folder, exist_ok=True)
            model_path = os.path.join(
                folder,
                f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth"
            )

            if cuda_index_list:
                net.module.save(model_path)
            else:
                net.save(model_path)

            logging.info(f"Saved: {model_path}")
