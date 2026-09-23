"""
step2_analyze_and_prune.py
──────────────────────────
Phase 2 of BN-gamma pruning pipeline for UltraFace (mb_tiny_fd).

CORRECTED VERSION — fixes incorrect tap indices that caused size mismatch errors.

The SSD forward pass uses source_layer_indexes = [8, 11, 13], which means:
    base_net[0:8]  → tap 1 (output of block 7,  64ch)  → header 0
    base_net[8:11] → tap 2 (output of block 10, 128ch) → header 1
    base_net[11:13]→ tap 3 (output of block 12, 256ch) → header 2
    extras output  → tap 4 (256ch)                     → header 3

So FIXED blocks (cannot prune output) are 7, 10, 12 — not 8, 11, 12 as previously believed.

What this does:
  1. Loads the sparsity-trained model from step1.
  2. Plots the gamma distribution.
  3. Given a threshold (or target % reduction), identifies which channels
     to prune in each prunable block.
  4. Rebuilds a new SSD with the surviving channel widths in the backbone.
  5. Copies the surviving weights into the new model.
  6. Saves the pruned model state dict ready for step3 fine-tuning.

Constraints respected automatically:
  - base_net[7]  output (64ch)  kept intact  → cls/reg header[0] input
  - base_net[10] output (128ch) kept intact  → cls/reg header[1] input
  - base_net[12] output (256ch) kept intact  → extras + cls/reg header[2] input
  - A minimum of 4 channels is always kept per layer (safety floor)
  - Channel counts are rounded down to nearest multiple of 4 (hardware friendly)

Usage:
    # First run with --analyze_only to see the gamma distribution
    python step2_analyze_and_prune.py --weights models/pruning/sparsity_trained.pth --analyze_only

    # Then run with a threshold to actually prune
    python step2_analyze_and_prune.py \
        --weights models/pruning/sparsity_trained.pth \
        --threshold 0.124

    # Or target a specific % of params to remove (script finds threshold automatically)
    python step2_analyze_and_prune.py \
        --weights models/pruning/sparsity_trained.pth \
        --target_reduction 0.10

python step2_analyze_and_prune.py --weights models/pruning/sparsity_trained.pth --target_reduction 0.35
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser()
parser.add_argument('--weights',          required=True)
parser.add_argument('--input_size',       default=320,   type=int)
parser.add_argument('--threshold',        default=None,  type=float,
                    help='Prune channels with |gamma| < threshold')
parser.add_argument('--target_reduction', default=None,  type=float,
                    help='Auto-find threshold to hit this fraction of param reduction (e.g. 0.10 = 10%%)')
parser.add_argument('--analyze_only',     action='store_true',
                    help='Just print/plot gamma stats, do not prune')
parser.add_argument('--output_dir',       default='models/pruning')
parser.add_argument('--plot',             action='store_true',
                    help='Save gamma histogram plot (requires matplotlib)')
args = parser.parse_args()

define_img_size(args.input_size)

from vision.ssd.config import fd_config
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# CORRECTED: tap points are at blocks 7, 10, 12 (per SSD forward with source_layer_indexes=[8,11,13])
PRUNABLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]
FIXED_OUTPUT_INDICES = {7: 64, 10: 128, 12: 256}
ORIGINAL_WIDTHS = [16, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 256, 256]
MIN_CHANNELS = 4


def get_block_bn_gammas(base_net):
    """
    For each prunable block, collect the gamma values of the POINTWISE BN
    (the one that controls the output channel count).

    conv_bn  (block 0):  has 1 BN at position .1  → output channels = 16
    conv_dw  (block 1+): has 2 BNs: .1 (depthwise) and .4 (pointwise)
                         We prune on .4 because that sets the output channel width.
    """
    block_gammas = {}
    for idx in PRUNABLE_INDICES:
        block = base_net[idx]
        bns = [(name, m) for name, m in block.named_modules()
               if isinstance(m, nn.BatchNorm2d)]
        # Last BN in the block = pointwise output BN
        _, output_bn = bns[-1]
        block_gammas[idx] = output_bn.weight.abs().detach().cpu()
    return block_gammas


def compute_surviving_channels(block_gammas, threshold):
    """
    For each block, return the indices of channels that survive (|gamma| >= threshold).
    Enforces minimum channel count and rounding to multiple of 4.
    """
    surviving = {}
    for idx, gammas in block_gammas.items():
        mask = gammas >= threshold
        keep_idx = mask.nonzero(as_tuple=True)[0]

        # Safety floor
        if len(keep_idx) < MIN_CHANNELS:
            keep_idx = gammas.topk(MIN_CHANNELS).indices.sort().values

        # Round down to multiple of 4
        n = len(keep_idx)
        n = max(MIN_CHANNELS, (n // 4) * 4)
        if n < len(keep_idx):
            top_n = gammas[keep_idx].topk(n).indices
            keep_idx = keep_idx[top_n.sort().values]

        surviving[idx] = keep_idx
    return surviving


def find_threshold_for_target(block_gammas, original_net, target_reduction):
    """Binary search for a threshold that hits ~target_reduction of param savings."""
    all_gammas = torch.cat(list(block_gammas.values()))
    lo, hi = 0.0, float(all_gammas.max())
    original_params = sum(p.numel() for p in original_net.parameters())
    target_params = original_params * (1.0 - target_reduction)

    mid = lo
    for _ in range(50):
        mid = (lo + hi) / 2
        surviving = compute_surviving_channels(block_gammas, mid)
        channel_map = {idx: len(v) for idx, v in surviving.items()}
        try:
            pruned_net = build_pruned_network(original_net, surviving, channel_map)
            pruned_params = sum(p.numel() for p in pruned_net.parameters())
            if pruned_params > target_params:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid

    return mid


def build_pruned_network(original_net, surviving_channels, new_channel_map):
    """
    Build a new SSD model with pruned channel widths in the backbone.

    new_channel_map: {block_idx: n_surviving_channels}
    Fixed blocks (7, 10, 12) always use their original widths.
    """
    new_widths = {}
    for idx in range(13):
        if idx in FIXED_OUTPUT_INDICES:
            new_widths[idx] = FIXED_OUTPUT_INDICES[idx]
        elif idx in new_channel_map:
            new_widths[idx] = new_channel_map[idx]
        else:
            new_widths[idx] = ORIGINAL_WIDTHS[idx]

    pruned_backbone = _build_pruned_backbone(new_widths)

    num_classes = 3
    from torch.nn import ModuleList, Sequential, Conv2d, ReLU
    from vision.ssd.mb_tiny_fd import SeperableConv2d
    from vision.ssd.ssd import SSD

    # Heads must match the FPN tap output channels exactly.
    # Tap channels are determined by the FIXED blocks (taps come BEFORE the slice end).
    tap1_ch  = new_widths[7]   # 64 always (fixed)
    tap2_ch  = new_widths[10]  # 128 always (fixed)
    tap3_ch  = new_widths[12]  # 256 always (fixed)
    extra_ch = 256             # extras output is fixed at 256

    extras = ModuleList([
        Sequential(
            Conv2d(tap3_ch, 64, kernel_size=1),
            ReLU(),
            SeperableConv2d(64, extra_ch, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])
    regression_headers = ModuleList([
        SeperableConv2d(tap1_ch,  3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(tap2_ch,  2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(tap3_ch,  2 * 4, kernel_size=3, padding=1),
        Conv2d(extra_ch, 3 * 4, kernel_size=3, padding=1),
    ])
    classification_headers = ModuleList([
        SeperableConv2d(tap1_ch,  3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(tap2_ch,  2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(tap3_ch,  2 * num_classes, kernel_size=3, padding=1),
        Conv2d(extra_ch, 3 * num_classes, kernel_size=3, padding=1),
    ])

    source_layer_indexes = [8, 11, 13]
    net = SSD(num_classes, pruned_backbone, source_layer_indexes,
              extras, classification_headers, regression_headers,
              is_test=False, config=fd_config, device='cpu')
    return net


def _build_pruned_backbone(new_widths):
    """Build an nn.Sequential backbone with the new (pruned) channel widths."""
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

    w = new_widths
    blocks = [
        conv_bn(3,     w[0],  2),
        conv_dw(w[0],  w[1],  1),
        conv_dw(w[1],  w[2],  2),
        conv_dw(w[2],  w[3],  1),
        conv_dw(w[3],  w[4],  2),
        conv_dw(w[4],  w[5],  1),
        conv_dw(w[5],  w[6],  1),
        conv_dw(w[6],  w[7],  1),   # tap1 — output FIXED at 64
        conv_dw(w[7],  w[8],  2),
        conv_dw(w[8],  w[9],  1),
        conv_dw(w[9],  w[10], 1),   # tap2 — output FIXED at 128
        conv_dw(w[10], w[11], 2),
        conv_dw(w[11], w[12], 1),   # tap3 — output FIXED at 256
    ]
    return nn.Sequential(*blocks)


def transfer_weights(original_net, pruned_net, surviving_channels):
    """
    Copy surviving weight slices from original into pruned model.
    """
    # Build the full surviving index map for all blocks
    all_surviving = {}
    for idx in range(13):
        if idx in surviving_channels:
            all_surviving[idx] = surviving_channels[idx]
        else:
            all_surviving[idx] = torch.arange(ORIGINAL_WIDTHS[idx])

    orig_base = original_net.base_net
    prnd_base = pruned_net.base_net

    for idx in range(13):
        orig_block = orig_base[idx]
        prnd_block = prnd_base[idx]
        out_idx = all_surviving[idx]
        in_idx  = all_surviving[idx - 1] if idx > 0 else torch.arange(3)

        if idx == 0:
            # conv_bn: [conv, bn, relu]
            prnd_block[0].weight.data = orig_block[0].weight.data[out_idx]
            _copy_bn(orig_block[1], prnd_block[1], out_idx)
        else:
            # conv_dw: [dw_conv, dw_bn, relu, pw_conv, pw_bn, relu]
            prnd_block[0].weight.data = orig_block[0].weight.data[in_idx]
            _copy_bn(orig_block[1], prnd_block[1], in_idx)
            prnd_block[3].weight.data = orig_block[3].weight.data[out_idx][:, in_idx]
            _copy_bn(orig_block[4], prnd_block[4], out_idx)

    # Copy heads unchanged (they connect to fixed-width tap outputs)
    def copy_module_state(src, dst):
        dst.load_state_dict(src.state_dict())

    copy_module_state(original_net.extras,                  pruned_net.extras)
    copy_module_state(original_net.classification_headers,  pruned_net.classification_headers)
    copy_module_state(original_net.regression_headers,      pruned_net.regression_headers)


def _copy_bn(orig_bn, prnd_bn, indices):
    prnd_bn.weight.data       = orig_bn.weight.data[indices]
    prnd_bn.bias.data         = orig_bn.bias.data[indices]
    prnd_bn.running_mean.data = orig_bn.running_mean.data[indices]
    prnd_bn.running_var.data  = orig_bn.running_var.data[indices]
    prnd_bn.num_batches_tracked.data = orig_bn.num_batches_tracked.data


def print_size_comparison(original_net, pruned_net, surviving_channels):
    logging.info('\n── Channel width comparison ────────────────────────────────')
    logging.info(f'  {"Block":<10} {"Original":>10} {"Pruned":>10} {"Removed":>10}')
    logging.info(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
    for idx in range(13):
        orig = ORIGINAL_WIDTHS[idx]
        if idx in surviving_channels:
            prnd = len(surviving_channels[idx])
            tag = '  ← pruned'
        elif idx in FIXED_OUTPUT_INDICES:
            prnd = orig
            tag = '  ← tap (fixed)'
        else:
            prnd = orig
            tag = ''
        logging.info(f'  bn[{idx:<2}]    {orig:>10} {prnd:>10} {orig-prnd:>10}{tag}')

    orig_p = sum(p.numel() for p in original_net.parameters())
    prnd_p = sum(p.numel() for p in pruned_net.parameters())
    reduction = (orig_p - prnd_p) / orig_p * 100
    logging.info(f'\n  Original params: {orig_p:,}')
    logging.info(f'  Pruned params:   {prnd_p:,}')
    logging.info(f'  Reduction:       {orig_p - prnd_p:,}  ({reduction:.1f}%)')

    # Print the --pruned_widths string for step 3
    width_list = []
    for idx in range(13):
        if idx in surviving_channels:
            width_list.append(len(surviving_channels[idx]))
        else:
            width_list.append(ORIGINAL_WIDTHS[idx])
    width_str = ','.join(str(w) for w in width_list)
    logging.info(f'\n  --pruned_widths "{width_str}"')


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info('Loading sparsity-trained model ...')
    net = create_mb_tiny_fd(num_classes=3, is_test=False, device='cpu')
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()

    block_gammas = get_block_bn_gammas(net.base_net)

    logging.info('\n── Gamma statistics per prunable block ─────────────────────')
    logging.info(f'  {"Block":<10} {"Channels":>8} {"Min":>8} {"Mean":>8} '
                 f'{"Max":>8} {"<0.01":>8} {"<0.05":>8}')
    logging.info(f'  {"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for idx, gammas in block_gammas.items():
        logging.info(f'  bn[{idx:<2}]    {len(gammas):>8} {gammas.min():>8.4f} '
                     f'{gammas.mean():>8.4f} {gammas.max():>8.4f} '
                     f'{(gammas<0.01).sum().item():>8} {(gammas<0.05).sum().item():>8}')

    all_g = torch.cat(list(block_gammas.values()))
    logging.info(f'\n  Global percentiles:')
    for p in [5, 10, 15, 20, 25]:
        k = max(1, int(p / 100 * len(all_g)))
        logging.info(f'    p{p:02d}: {all_g.kthvalue(k)[0]:.4f}')

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.hist(all_g.numpy(), bins=100, edgecolor='black')
            plt.xlabel('|gamma|')
            plt.ylabel('count')
            plt.title('BN Gamma Distribution (prunable blocks)')
            plt.axvline(x=0.05, color='r', linestyle='--', label='threshold=0.05')
            plt.legend()
            plot_path = os.path.join(args.output_dir, 'gamma_distribution.png')
            plt.savefig(plot_path, dpi=120)
            logging.info(f'Plot saved to {plot_path}')
        except ImportError:
            logging.warning('matplotlib not available, skipping plot')

    if args.analyze_only:
        logging.info('\nAnalysis complete. Re-run without --analyze_only to prune.')
        sys.exit(0)

    if args.threshold is not None:
        threshold = args.threshold
        logging.info(f'\nUsing manual threshold: {threshold}')
    elif args.target_reduction is not None:
        logging.info(f'\nSearching for threshold to achieve {args.target_reduction*100:.1f}% reduction ...')
        threshold = find_threshold_for_target(block_gammas, net, args.target_reduction)
        logging.info(f'Found threshold: {threshold:.4f}')
    else:
        logging.error('Provide --threshold or --target_reduction (or --analyze_only)')
        sys.exit(1)

    surviving = compute_surviving_channels(block_gammas, threshold)
    new_channel_map = {idx: len(v) for idx, v in surviving.items()}

    logging.info('\n── Surviving channels ───────────────────────────────────────')
    for idx, keep_idx in surviving.items():
        orig = block_gammas[idx].shape[0]
        logging.info(f'  bn[{idx}]: {orig} → {len(keep_idx)} channels kept')

    logging.info('\nBuilding pruned network ...')
    pruned_net = build_pruned_network(net, surviving, new_channel_map)
    logging.info('Transferring weights ...')
    transfer_weights(net, pruned_net, surviving)

    print_size_comparison(net, pruned_net, surviving)

    pruned_net.eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        conf, loc = pruned_net(dummy)
    logging.info(f'\nForward pass OK — conf: {conf.shape}  loc: {loc.shape}')

    out_path = os.path.join(args.output_dir, 'pruned_model.pth')
    pruned_net.save(out_path)
    logging.info(f'\nPruned model saved → {out_path}')
    logging.info('Next: run step3_finetune.py')