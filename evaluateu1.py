"""
evaluate.py
-----------
Evaluates a trained pruned .pth gesture detection model on the HaGRID test set.

Reports:
  • Per-class Detection Accuracy  (top detection class matches GT label)
  • mAP @ IoU 0.5                 (proper VOC-style with GT bounding boxes)
  • Confusion matrix

Changes vs previous version:
  - GT bounding boxes loaded from JSON for proper IoU-based TP/FP matching
  - prob_threshold=0.01 for inference so all detections enter mAP calculation
  - Confidence threshold (0.5) only used for per-image accuracy / confusion matrix
  - All-point AP interpolation (VOC 2010+) instead of 11-point

Expected layout:
    dataset/
    ├── palm_test.json
    ├── two_test.json
    └── test/
        ├── palm/
        └── two/

Usage:
    python evaluate.py --dataset_dir /path/to/dataset --model_path /path/to/model.pth --device cuda
"""

import json
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

# ── Project imports ───────────────────────────────────────────────────────────
from vision.ssd.config import fd_config as config
from vision.ssd.ssd import SSD
from vision.nn.mb_tiny_pruned import Mb_Tiny_Pruned
from torch.nn import Conv2d, Sequential, ModuleList, ReLU

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["background", "palm", "two"]
GESTURE_NAMES  = ["palm", "two"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IOU_THRESHOLD  = 0.5    # IoU threshold for TP/FP decision in mAP
ACC_THRESHOLD  = 0.5    # confidence threshold for accuracy / confusion matrix
MAP_THRESHOLD  = 0.01   # low threshold so all detections enter mAP ranking

# JSON filenames — adjust if yours differ
JSON_NAMES = {
    "palm": "palm_test.json",
    "two":  "two_test.json",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Model ─────────────────────────────────────────────────────────────────────
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return Sequential(
        Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
        ReLU(),
        Conv2d(in_channels, out_channels, 1),
    )


def build_model(num_classes: int, device: str) -> SSD:
    base_net   = Mb_Tiny_Pruned()
    base_model = base_net.model

    source_layer_indexes = [8, 11, 13]
    extras = ModuleList([
        Sequential(
            Conv2d(256, 54, kernel_size=1), ReLU(),
            SeperableConv2d(54, 217, kernel_size=3, stride=2, padding=1), ReLU()
        )
    ])
    regression_headers = ModuleList([
        SeperableConv2d(64,  3 * 4,           kernel_size=3, padding=1),
        SeperableConv2d(128, 2 * 4,           kernel_size=3, padding=1),
        SeperableConv2d(256, 2 * 4,           kernel_size=3, padding=1),
        Conv2d(217,          3 * 4,           kernel_size=3, padding=1),
    ])
    classification_headers = ModuleList([
        SeperableConv2d(64,  3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(128, 2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(256, 2 * num_classes, kernel_size=3, padding=1),
        Conv2d(217,          3 * num_classes, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, base_model, source_layer_indexes,
               extras, classification_headers, regression_headers,
               is_test=True, config=config, device=device)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes.
    Both in [x1, y1, x2, y2] format, normalised to [0, 1].
    """
    xA = max(box_a[0], box_b[0]);  yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]);  yB = min(box_a[3], box_b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hagrid_bbox_to_xyxy(bbox):
    """
    HaGRID stores bboxes as [x_center, y_center, width, height] normalised.
    Convert to [x1, y1, x2, y2] normalised.
    """
    cx, cy, w, h = bbox
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def compute_ap_allpoint(recalls, precisions):
    """
    All-point interpolation AP (VOC 2010+).
    More accurate than 11-point; standard in modern papers.
    """
    recalls    = np.concatenate(([0.0], recalls,    [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision monotonically decreasing right to left
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # Area under curve at recall change points
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def preprocess(image_path: Path, image_size: list) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[0], image_size[1]))
    return img


def uuid_from_filename(stem: str) -> str:
    """Extract UUID from plain or Roboflow-style filename stem."""
    return stem.split("_jpg.rf.")[0] if "_jpg.rf." in stem else stem


# ── Load GT boxes from JSON ───────────────────────────────────────────────────
def load_gt(dataset_dir: Path) -> dict:
    """
    Returns:
        gt_map  dict:  uuid -> {"class_idx": int, "boxes": [[x1,y1,x2,y2], ...]}
                       boxes normalised [0,1] in x1y1x2y2 format
    """
    gt_map = {}
    for gesture, json_name in JSON_NAMES.items():
        json_path = dataset_dir / json_name
        if not json_path.exists():
            print(f"  [WARN] {json_name} not found — GT boxes unavailable for {gesture}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        cls_idx = CLASS_NAMES.index(gesture)
        for uuid, entry in data.items():
            boxes_xyxy = [hagrid_bbox_to_xyxy(b) for b in entry.get("bboxes", [])]
            gt_map[uuid] = {
                "class_idx": cls_idx,
                "boxes":     boxes_xyxy,
            }

    print(f"  Loaded GT for {len(gt_map):,} images from JSON")
    return gt_map


# ── mAP (proper IoU-based, all-point) ────────────────────────────────────────
def compute_map(all_results: list) -> dict:
    """
    Proper VOC-style mAP with IoU-based TP/FP matching.

    all_results: list of dicts:
        gt_class_idx  int
        gt_boxes      list of [x1,y1,x2,y2] normalised
        pred_boxes    np.ndarray (N,4) normalised
        pred_scores   np.ndarray (N,)
        pred_classes  np.ndarray (N,) 1-based
    """
    aps = {}

    for cls_idx in range(1, len(GESTURE_NAMES) + 1):
        cls_name   = CLASS_NAMES[cls_idx]
        detections = []   # (score, is_tp)
        n_gt       = 0

        for res in all_results:
            is_gt_class  = (res["gt_class_idx"] == cls_idx)
            gt_boxes_cls = res["gt_boxes"] if is_gt_class else []

            if is_gt_class:
                # Count GT boxes; treat as 1 GT if bboxes were missing from JSON
                n_gt += len(gt_boxes_cls) if gt_boxes_cls else 1

            # Filter predictions to this class
            mask   = res["pred_classes"] == cls_idx
            scores = res["pred_scores"][mask]
            boxes  = res["pred_boxes"][mask]

            matched_gt = set()
            order      = np.argsort(-scores)  # descending confidence

            for i in order:
                box   = boxes[i]
                score = float(scores[i])

                # Greedy match to best available GT box by IoU
                best_iou = 0.0
                best_j   = -1
                for j, gt_box in enumerate(gt_boxes_cls):
                    if j in matched_gt:
                        continue
                    iou_val = compute_iou(box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j   = j

                if best_iou >= IOU_THRESHOLD and best_j >= 0:
                    detections.append((score, 1))   # TP
                    matched_gt.add(best_j)
                else:
                    detections.append((score, 0))   # FP

        if n_gt == 0:
            aps[cls_name] = float("nan")
            continue
        if not detections:
            aps[cls_name] = 0.0
            continue

        detections.sort(key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in detections]).astype(float)
        fp_cum = np.cumsum([1 - d[1] for d in detections]).astype(float)

        recalls    = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        aps[cls_name] = compute_ap_allpoint(recalls, precisions)

    return aps


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(dataset_dir: str, model_path: str, device: str):
    dataset_dir = Path(dataset_dir)
    device      = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load GT bounding boxes from JSON
    print("\n[1/3] Loading ground truth …")
    gt_map = load_gt(dataset_dir)

    # Build and load model
    print("\n[2/3] Loading model …")
    config.define_img_size(320)
    num_classes = len(CLASS_NAMES)
    net = build_model(num_classes, str(device))
    net.load(model_path)
    net.to(device)
    net.eval()
    print(f"  Loaded: {model_path}")

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

    # ── Inference loop ────────────────────────────────────────────────────────
    print("\n[3/3] Running inference …\n")
    test_dir    = dataset_dir / "test"
    all_results = []
    correct     = defaultdict(int)
    total       = defaultdict(int)
    conf_matrix = np.zeros((2, 3), dtype=int)  # rows=GT, cols=pred (+no-detect)

    for gt_idx, gesture in enumerate(GESTURE_NAMES):
        cls_dir = test_dir / gesture
        if not cls_dir.exists():
            print(f"  [WARN] {cls_dir} not found — skipping")
            continue

        image_paths = [p for p in cls_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTENSIONS]
        print(f"  {gesture}: {len(image_paths):,} images")

        gt_class_idx = CLASS_NAMES.index(gesture)

        #ADDED THIS
        matched = 0
        for img_path in image_paths[:20]:   # check first 20 images
            uuid     = uuid_from_filename(img_path.stem)
            gt_entry = gt_map.get(uuid, {})
            gt_boxes = gt_entry.get("boxes", [])
            if gt_boxes:
                matched += 1
            print(f"  {img_path.stem[:40]}  →  uuid={uuid[:20]}  gt_boxes={len(gt_boxes)}")
        print(f"\nMatched {matched}/20 images to GT boxes")
        #END OF ADD

        for img_path in image_paths:
            uuid     = uuid_from_filename(img_path.stem)
            gt_entry = gt_map.get(uuid, {})
            gt_boxes = gt_entry.get("boxes", [])

            try:
                img = preprocess(img_path, config.image_size)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            # Use low threshold so all detections enter mAP
            with torch.no_grad():
                boxes, labels, scores = predictor.predict(
                    img, top_k=10, prob_threshold=MAP_THRESHOLD)

            boxes  = boxes.numpy()  if torch.is_tensor(boxes)  else np.array(boxes)
            labels = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
            scores = scores.numpy() if torch.is_tensor(scores) else np.array(scores)

            # Normalise predicted boxes to [0,1]
            W, H       = config.image_size[0], config.image_size[1]
            norm_boxes = boxes / np.array([W, H, W, H], dtype=float) \
                         if len(boxes) else np.zeros((0, 4))

            total[gesture] += 1

            # ── Per-image accuracy (ACC_THRESHOLD filter) ─────────────────
            high_conf = scores >= ACC_THRESHOLD
            hc_labels = labels[high_conf]
            hc_scores = scores[high_conf]

            if len(hc_labels) == 0:
                conf_matrix[gt_idx, 2] += 1          # no detection
            else:
                best      = np.argmax(hc_scores)
                pred_cls  = int(hc_labels[best])
                pred_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else "unknown"

                if pred_cls == gt_class_idx:
                    correct[gesture] += 1
                    conf_matrix[gt_idx, gt_idx] += 1
                else:
                    pred_row = GESTURE_NAMES.index(pred_name) \
                               if pred_name in GESTURE_NAMES else 2
                    conf_matrix[gt_idx, pred_row] += 1

            # ── Collect for mAP ───────────────────────────────────────────
            all_results.append({
                "gt_class_idx": gt_class_idx,
                "gt_boxes":     gt_boxes,
                "pred_boxes":   norm_boxes,
                "pred_scores":  scores,
                "pred_classes": labels,
            })

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  RESULTS")
    print("═" * 58)

    print(f"\n── Per-class Detection Accuracy  (conf ≥ {ACC_THRESHOLD:.2f}) ──────")
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
        print(f"  {'TOTAL':>6}: {overall_correct:>5} / {overall_total:>5}"
              f"  =  {overall_correct / overall_total * 100:.2f}%")

    print(f"\n── mAP @ IoU {IOU_THRESHOLD}  (all-point interpolation) ────────────")
    aps = compute_map(all_results)
    for cls_name, ap in aps.items():
        if np.isnan(ap):
            print(f"  {cls_name:>6}: N/A (no GT samples)")
        else:
            print(f"  {cls_name:>6}: {ap:.4f}")
    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    if valid_aps:
        print(f"  {'mAP':>6}: {np.mean(valid_aps):.4f}")

    print("\n── Confusion Matrix ──────────────────────────────────────")
    print(f"  {'':>10}  {'pred:palm':>10}  {'pred:two':>10}  {'no detect':>10}")
    for i, gesture in enumerate(GESTURE_NAMES):
        row = conf_matrix[i]
        print(f"  {'gt:'+gesture:>10}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")

    print("\n" + "═" * 58)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   type=str,   required=True)
    parser.add_argument("--model_path",    type=str,   required=True)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--acc_threshold", type=float, default=0.5,
                        help="Confidence threshold for accuracy/confusion matrix (default: 0.5)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for TP/FP matching in mAP (default: 0.5)")
    args = parser.parse_args()

    ACC_THRESHOLD = args.acc_threshold
    IOU_THRESHOLD = args.iou_threshold
    evaluate(args.dataset_dir, args.model_path, args.device)
