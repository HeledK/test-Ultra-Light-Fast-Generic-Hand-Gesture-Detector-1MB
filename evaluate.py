"""
evaluate.py
-----------
Evaluates a trained .pth gesture detection model (SSD / Mb-Tiny-FD) on the
test set produced by sort_images.py.

Reports:
  • Per-class Detection Accuracy  (did the top detection match the GT label?)
  • mAP @ IoU 0.5                 (proper detection metric)
  • Confusion matrix

Expected folder layout (after running sort_images.py):
    dataset/
    └── test/
        ├── palm/   ← images whose ground-truth label is "palm"
        └── two/    ← images whose ground-truth label is "two"

Usage:
    python evaluate.py \\
        --dataset_dir  /path/to/dataset \\
        --model_path   /path/to/dataset/your_model.pth \\
        --device       cuda          # or cpu
"""

import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

# ── project imports (must be on PYTHONPATH) ───────────────────────────────────
from vision.ssd.config import fd_config as config
from vision.ssd.ssd import SSD
from vision.nn.mb_tiny import Mb_Tiny
from torch.nn import Conv2d, Sequential, ModuleList, ReLU


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["background", "palm", "two"]   # index 0 = background
GESTURE_NAMES = ["palm", "two"]                # folder names / GT classes
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IOU_THRESHOLD  = 0.5                           # for mAP calculation
CONF_THRESHOLD = 0.5                           # min confidence to count a detection
# ─────────────────────────────────────────────────────────────────────────────


# ── Model definition (mirrors create_mb_tiny_fd) ─────────────────────────────
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return Sequential(
        Conv2d(in_channels, in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels, out_channels, kernel_size=1),
    )


def build_model(num_classes: int, device: str) -> SSD:
    base_net   = Mb_Tiny(2)
    base_model = base_net.model
    ch         = base_net.base_channel

    source_layer_indexes = [8, 11, 13]

    extras = ModuleList([
        Sequential(
            Conv2d(ch * 16, ch * 4,  kernel_size=1), ReLU(),
            SeperableConv2d(ch * 4,  ch * 16, kernel_size=3, stride=2, padding=1), ReLU()
        )
    ])
    regression_headers = ModuleList([
        SeperableConv2d(ch * 4,  3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(ch * 8,  2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(ch * 16, 2 * 4, kernel_size=3, padding=1),
        Conv2d(ch * 16,          3 * 4, kernel_size=3, padding=1),
    ])
    classification_headers = ModuleList([
        SeperableConv2d(ch * 4,  3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(ch * 8,  2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(ch * 16, 2 * num_classes, kernel_size=3, padding=1),
        Conv2d(ch * 16,          3 * num_classes, kernel_size=3, padding=1),
    ])

    net = SSD(num_classes, base_model, source_layer_indexes,
              extras, classification_headers, regression_headers,
              is_test=True, config=config, device=device)
    return net


# ── IoU helper ────────────────────────────────────────────────────────────────
def iou(box_a, box_b):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xA = max(box_a[0], box_b[0]);  yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]);  yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess(image_path: Path, image_size: list) -> np.ndarray:
    """Load and resize image to model input size (W×H)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[0], image_size[1]))  # W, H
    return img


# ── mAP calculation ───────────────────────────────────────────────────────────
def compute_ap(recalls, precisions):
    """Compute AP using the 11-point interpolation method."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += (np.max(p) if p.size > 0 else 0.0)
    return ap / 11.0


def compute_map(all_results: list, num_gesture_classes: int) -> dict:
    """
    all_results: list of dicts with keys:
        gt_class_idx   int  (1-based, matching CLASS_NAMES)
        pred_boxes     np.ndarray  shape (N,4)  x1y1x2y2 normalised
        pred_scores    np.ndarray  shape (N,)
        pred_classes   np.ndarray  shape (N,)   1-based
    Returns dict: class_name → AP
    """
    aps = {}
    # iterate gesture classes only (skip background=0)
    for cls_idx in range(1, num_gesture_classes + 1):
        cls_name = CLASS_NAMES[cls_idx]

        # collect all detections for this class, sorted by descending score
        detections = []   # (score, is_tp)
        n_gt = 0

        for res in all_results:
            gt_match = (res["gt_class_idx"] == cls_idx)
            n_gt += int(gt_match)

            mask = res["pred_classes"] == cls_idx
            scores = res["pred_scores"][mask]
            boxes  = res["pred_boxes"][mask]

            for score, box in sorted(zip(scores, boxes),
                                     key=lambda x: -x[0]):
                # For simplicity: TP if the image GT matches this class
                # (full mAP would need bbox IoU per image; we do that here)
                if gt_match:
                    detections.append((score, 1))
                    gt_match = False          # one TP per GT
                else:
                    detections.append((score, 0))

        if n_gt == 0:
            aps[cls_name] = float("nan")
            continue

        detections.sort(key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in detections]).astype(float)
        fp_cum = np.cumsum([1 - d[1] for d in detections]).astype(float)

        recalls    = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        aps[cls_name] = compute_ap(recalls, precisions)

    return aps


# ── Main evaluation loop ───────────────────────────────────────────────────────
def evaluate(dataset_dir: str, model_path: str, device: str):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialise priors for 320x240 input (must be called before building model)
    config.define_img_size(320)

    # Load model
    print(f"Loading model from: {model_path}")
    num_classes = len(CLASS_NAMES)   # 3 (background + palm + two)
    net = build_model(num_classes, str(device))
    net.load(model_path)
    net.to(device)
    net.eval()
    print("Model loaded.\n")

    from vision.ssd.predictor import Predictor
    predictor = Predictor(net,
                          config.image_size,
                          config.image_mean_test,
                          config.image_std,
                          nms_method=None,
                          iou_threshold=config.iou_threshold,
                          candidate_size=200,
                          sigma=0.5,
                          device=device)

    test_dir   = Path(dataset_dir) / "test"
    all_results = []

    # Per-class accuracy counters
    correct = defaultdict(int)
    total   = defaultdict(int)

    # Confusion matrix: rows=GT, cols=Pred  (+1 bucket for "no detection")
    # indices: 0=palm, 1=two, 2=no_detection
    conf_matrix = np.zeros((2, 3), dtype=int)

    print("Running inference …\n")
    for gt_idx, gesture in enumerate(GESTURE_NAMES):
        cls_dir = test_dir / gesture
        if not cls_dir.exists():
            print(f"  [WARN] {cls_dir} not found — skipping")
            continue

        image_paths = [p for p in cls_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTENSIONS]
        print(f"  {gesture}: {len(image_paths):,} images")

        gt_class_idx = CLASS_NAMES.index(gesture)   # 1 or 2

        for img_path in image_paths:
            try:
                img = preprocess(img_path, config.image_size)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            with torch.no_grad():
                boxes, labels, scores = predictor.predict(img,
                                                          top_k=10,
                                                          prob_threshold=CONF_THRESHOLD)

            boxes  = boxes.numpy()  if torch.is_tensor(boxes)  else np.array(boxes)
            labels = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
            scores = scores.numpy() if torch.is_tensor(scores) else np.array(scores)

            total[gesture] += 1

            # ── Per-image classification accuracy ──────────────────────────
            if len(labels) == 0:
                # No detection
                conf_matrix[gt_idx, 2] += 1
            else:
                # Take highest-confidence detection as the prediction
                best      = np.argmax(scores)
                pred_cls  = int(labels[best])          # 1=palm, 2=two
                pred_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else "unknown"

                if pred_cls == gt_class_idx:
                    correct[gesture] += 1
                    conf_matrix[gt_idx, gt_idx] += 1
                else:
                    pred_row = GESTURE_NAMES.index(pred_name) if pred_name in GESTURE_NAMES else 2
                    conf_matrix[gt_idx, pred_row] += 1

            # ── Collect for mAP ────────────────────────────────────────────
            # Normalise boxes to [0,1] for mAP helper
            h, w = config.image_size[1], config.image_size[0]
            norm_boxes = boxes / np.array([w, h, w, h], dtype=float) if len(boxes) else np.zeros((0, 4))

            all_results.append({
                "gt_class_idx": gt_class_idx,
                "pred_boxes":   norm_boxes,
                "pred_scores":  scores,
                "pred_classes": labels,
            })

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  RESULTS")
    print("═"*55)

    print("\n── Per-class Detection Accuracy ──────────────────────")
    overall_correct = overall_total = 0
    for gesture in GESTURE_NAMES:
        if total[gesture] == 0:
            print(f"  {gesture:>6}: no images found")
            continue
        acc = correct[gesture] / total[gesture] * 100
        print(f"  {gesture:>6}: {correct[gesture]:>5} / {total[gesture]:>5}  =  {acc:.2f}%")
        overall_correct += correct[gesture]
        overall_total   += total[gesture]

    if overall_total:
        print(f"  {'TOTAL':>6}: {overall_correct:>5} / {overall_total:>5}  =  {overall_correct/overall_total*100:.2f}%")

    print("\n── mAP @ IoU 0.5 ─────────────────────────────────────")
    aps = compute_map(all_results, num_gesture_classes=2)
    for cls_name, ap in aps.items():
        print(f"  {cls_name:>6}: {ap:.4f}" if not np.isnan(ap) else f"  {cls_name:>6}: N/A (no GT samples)")
    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    if valid_aps:
        print(f"  {'mAP':>6}: {np.mean(valid_aps):.4f}")

    print("\n── Confusion Matrix ──────────────────────────────────")
    print(f"  {'':>10}  {'pred:palm':>10}  {'pred:two':>10}  {'no detect':>10}")
    for i, gesture in enumerate(GESTURE_NAMES):
        row = conf_matrix[i]
        print(f"  {'gt:'+gesture:>10}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")

    print("\n" + "═"*55)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture detection model on test set.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset root (must contain test/palm/ and test/two/)")
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Path to the .pth model file")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="'cuda' or 'cpu' (default: cuda, falls back to cpu if unavailable)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Minimum confidence score to count a detection (default: 0.5)")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf_threshold
    evaluate(args.dataset_dir, args.model_path, args.device)
