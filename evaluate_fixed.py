"""
evaluate.py
-----------
Evaluates a trained .pth gesture detection model (SSD / Mb-Tiny-FD) on the
test set.

Changes from original:
  - GT boxes loaded from JSON (palm_test.json, two_test.json)
  - NMS applied before mAP (removes duplicate detections per image)
  - mAP uses proper IoU box matching (not just class match)
  - mAP uses all-point interpolation (matches evaluate_tflite.py)

Reports:
  • Per-class Detection Accuracy  (conf >= CONF_THRESHOLD)
  • mAP @ IoU 0.5                 (all-point interpolation, VOC 2010+)
  • Confusion matrix

Usage:
    python evaluate.py \
        --dataset_dir  /path/to/dataset \
        --model_path   /path/to/your_model.pth \
        --device       cuda
"""

import os
import json
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

from vision.ssd.config import fd_config as config
from vision.ssd.ssd import SSD
from vision.nn.mb_tiny import Mb_Tiny
from torch.nn import Conv2d, Sequential, ModuleList, ReLU

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["background", "palm", "two"]
GESTURE_NAMES  = ["palm", "two"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.3    # matches evaluate_tflite.py
MAP_THRESHOLD  = 0.01   # min score to enter mAP ranking

JSON_NAMES = {
    "palm": "palm_test.json",
    "two":  "two_test.json",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Model definition ──────────────────────────────────────────────────────────
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return Sequential(
        Conv2d(in_channels, in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels, out_channels, kernel_size=1),
    )


def build_model(num_classes, device):
    base_net   = Mb_Tiny(2)
    base_model = base_net.model
    ch         = base_net.base_channel

    extras = ModuleList([
        Sequential(
            Conv2d(ch * 16, ch * 4,  kernel_size=1), ReLU(),
            SeperableConv2d(ch * 4, ch * 16, kernel_size=3, stride=2, padding=1), ReLU()
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

    net = SSD(num_classes, base_model, [8, 11, 13],
              extras, classification_headers, regression_headers,
              is_test=True, config=config, device=device)
    return net


# ── GT loading ────────────────────────────────────────────────────────────────
def hagrid_bbox_to_xyxy(bbox):
    """HaGRID [x1, y1, w, h] normalised → [x1, y1, x2, y2] normalised."""
    x1, y1, w, h = bbox
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(1.0, x1 + w),
        min(1.0, y1 + h),
    ]


def load_gt(dataset_dir):
    """Load GT boxes from JSON files. Returns {uuid: {class_idx, boxes}}."""
    gt_map = {}
    for gesture, json_name in JSON_NAMES.items():
        json_path = Path(dataset_dir) / json_name
        if not json_path.exists():
            print(f"  [WARN] {json_name} not found — GT boxes unavailable for {gesture}")
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        cls_idx = CLASS_NAMES.index(gesture)
        for uuid, entry in data.items():
            boxes = [hagrid_bbox_to_xyxy(b) for b in entry.get("bboxes", [])]
            gt_map[uuid] = {"class_idx": cls_idx, "boxes": boxes}
    print(f"  Loaded GT for {len(gt_map):,} images")
    return gt_map


# ── IoU ───────────────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0]);  yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]);  yB = min(box_a[3], box_b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── NMS ───────────────────────────────────────────────────────────────────────
def apply_nms(boxes, scores, labels, nms_threshold=NMS_THRESHOLD):
    """
    Apply NMS per class. Returns filtered (boxes, scores, labels).
    boxes: np.ndarray (N, 4) normalised x1y1x2y2
    scores: np.ndarray (N,)
    labels: np.ndarray (N,) 1-based class indices
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    keep_boxes  = []
    keep_scores = []
    keep_labels = []

    for cls_idx in range(1, len(CLASS_NAMES)):
        mask = labels == cls_idx
        if not mask.any():
            continue

        cls_boxes  = boxes[mask]
        cls_scores = scores[mask]

        boxes_xywh = []
        for b in cls_boxes:
            x1, y1, x2, y2 = b
            boxes_xywh.append([float(x1), float(y1),
                                float(x2 - x1), float(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            cls_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=nms_threshold
        )

        if len(indices) == 0:
            continue

        indices = indices.flatten()
        for i in indices:
            keep_boxes.append(cls_boxes[i])
            keep_scores.append(cls_scores[i])
            keep_labels.append(cls_idx)

    if not keep_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return (np.array(keep_boxes),
            np.array(keep_scores),
            np.array(keep_labels, dtype=int))


# ── mAP ───────────────────────────────────────────────────────────────────────
def compute_ap_allpoint(recalls, precisions):
    """All-point interpolation AP — VOC 2010+ standard."""
    recalls    = np.concatenate(([0.0], recalls,    [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def compute_map(all_results):
    """
    all_results: list of dicts with keys:
        gt_class_idx  int           (1-based)
        gt_boxes      list of [x1,y1,x2,y2] normalised
        pred_boxes    np.ndarray    (N, 4)
        pred_scores   np.ndarray    (N,)   per-detection score for that class
        pred_labels   np.ndarray    (N,)   1-based class index
    """
    aps = {}
    for cls_idx in range(1, len(GESTURE_NAMES) + 1):
        cls_name   = CLASS_NAMES[cls_idx]
        detections = []
        n_gt       = 0

        for res in all_results:
            is_gt_cls    = (res["gt_class_idx"] == cls_idx)
            gt_boxes_cls = res["gt_boxes"] if is_gt_cls else []
            if is_gt_cls:
                n_gt += len(gt_boxes_cls) if gt_boxes_cls else 1

            # Only look at detections for this class
            mask = res["pred_labels"] == cls_idx
            scores = res["pred_scores"][mask]
            boxes  = res["pred_boxes"][mask]

            matched_gt = set()
            order = np.argsort(-scores)

            for i in order:
                score = float(scores[i])
                if score < MAP_THRESHOLD:
                    continue
                box = boxes[i]

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


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_path, image_size):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[0], image_size[1]))
    return img


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(dataset_dir, model_path, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config.define_img_size(320)

    # Load GT boxes from JSON
    print("\n[1/3] Loading ground truth ...")
    gt_map = load_gt(dataset_dir)

    # Load model
    print("\n[2/3] Loading model ...")
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
                          nms_method=None,        # we apply NMS ourselves
                          iou_threshold=1.0,      # effectively disable built-in NMS
                          candidate_size=500,
                          sigma=0.5,
                          device=device)

    print("\n[3/3] Running inference ...")
    test_dir    = Path(dataset_dir) / "test"
    all_results = []
    correct     = defaultdict(int)
    total       = defaultdict(int)
    conf_matrix = np.zeros((2, 3), dtype=int)

    for gt_idx, gesture in enumerate(GESTURE_NAMES):
        cls_dir = test_dir / gesture
        if not cls_dir.exists():
            print(f"  [WARN] {cls_dir} not found — skipping")
            continue

        image_paths = [p for p in cls_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTENSIONS]
        print(f"  {gesture}: {len(image_paths):,} images")
        gt_class_idx = CLASS_NAMES.index(gesture)

        for img_path in image_paths:
            uuid     = img_path.stem
            gt_entry = gt_map.get(uuid, {})
            gt_boxes = gt_entry.get("boxes", [])

            try:
                img = preprocess(img_path, config.image_size)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            with torch.no_grad():
                boxes, labels, scores = predictor.predict(
                    img, top_k=50, prob_threshold=MAP_THRESHOLD)

            boxes  = boxes.numpy()  if torch.is_tensor(boxes)  else np.array(boxes)
            labels = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
            scores = scores.numpy() if torch.is_tensor(scores) else np.array(scores)

            # Normalise boxes to [0,1]
            h, w = config.image_size[1], config.image_size[0]
            if len(boxes):
                norm_boxes = boxes / np.array([w, h, w, h], dtype=float)
                norm_boxes = np.clip(norm_boxes, 0.0, 1.0)
            else:
                norm_boxes = np.zeros((0, 4))

            # Apply NMS per class before mAP and accuracy
            norm_boxes, scores, labels = apply_nms(norm_boxes, scores, labels)

            total[gesture] += 1

            # ── Per-image classification accuracy ─────────────────────────
            high_conf = scores >= CONF_THRESHOLD if len(scores) else np.array([])
            if not high_conf.any():
                conf_matrix[gt_idx, 2] += 1   # no detection
            else:
                hc_scores = scores[high_conf]
                hc_labels = labels[high_conf]
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
                "pred_labels":  labels,
            })

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  RESULTS")
    print("═" * 58)

    print(f"\n── Per-class Detection Accuracy  (conf ≥ {CONF_THRESHOLD}) ────────")
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

    print(f"\n── mAP @ IoU {IOU_THRESHOLD}  (all-point interpolation) ───────────────")
    aps = compute_map(all_results)
    for cls_name, ap in aps.items():
        if np.isnan(ap):
            print(f"  {cls_name:>6}: N/A")
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
    parser.add_argument("--dataset_dir",    type=str, required=True)
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf_threshold
    evaluate(args.dataset_dir, args.model_path, args.device)