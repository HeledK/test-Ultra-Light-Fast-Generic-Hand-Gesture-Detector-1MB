"""
evaluate_tflite.py
------------------
Evaluates the INT8 quantized TFLite gesture detection model on the HaGRID
test set. Use this to measure accuracy loss from quantization vs the .pth.

Output format per detection row: [bg, palm, two, left, top, right, bottom]
  - bg, palm, two : class scores (sum ~1.0)
  - left, top, right, bottom : normalised bounding box [0, 1]

Reports:
  • Per-class Detection Accuracy  (conf >= ACC_THRESHOLD)
  • mAP @ IoU 0.5                 (all-point interpolation, VOC 2010+)
  • Confusion matrix

Expected layout:
    dataset/
    ├── palm_test.json
    ├── two_test.json
    └── test/
        ├── palm/
        └── two/

Usage:
    python evaluate_tflite.py \
        --dataset_dir  /path/to/dataset \
        --model_path   /path/to/model_int8.tflite \
        --acc_threshold 0.5
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        raise ImportError("Install tensorflow or tflite-runtime: pip install tensorflow")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["background", "palm", "two"]
GESTURE_NAMES = ["palm", "two"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INPUT_SIZE     = (320, 240)   # W, H  — confirmed from get_input_details()  #either 320,240  or 128,128
IOU_THRESHOLD  = 0.5          # for mAP TP/FP matching
ACC_THRESHOLD  = 0.5          # for per-image accuracy & confusion matrix
MAP_THRESHOLD  = 0.01         # min score to enter mAP ranking

# Input quantization params (from get_input_details())
INP_SCALE      = 0.0078125
INP_ZERO_POINT = -1

# Output quantization params (from get_output_details())
OUT_SCALE      = 0.003921568859368563
OUT_ZERO_POINT = -128

# JSON filenames
JSON_NAMES = {
    "palm": "palm_test.json",
    "two":  "two_test.json",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── TFLite helpers ────────────────────────────────────────────────────────────
def load_interpreter(model_path: str):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    return interp, inp_details, out_details


def preprocess(image_path: Path) -> np.ndarray:
    """Load, resize, and quantize image to INT8 for model input."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #maybe remove
    img = cv2.resize(img, INPUT_SIZE).astype(np.float32)
    # Normalize first (must match representative_dataset in ultraslim.py)
    img = (img - 127.0) / 128.0
    # Quantize: float32 -> int8
    img_q = np.clip(
        np.round(img / INP_SCALE + INP_ZERO_POINT), -128, 127
    ).astype(np.int8)
    return img_q[np.newaxis]  # add batch dim → (1, 128, 128, 3)


def run_inference(interp, inp_details, out_details, img_q: np.ndarray) -> np.ndarray:
    """Run inference and dequantize output to float32."""
    interp.set_tensor(inp_details['index'], img_q)
    interp.invoke()
    raw = interp.get_tensor(out_details['index'])          # int8 (172, 7)
    return (raw.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE  # float (172, 7)


# ── GT loading ────────────────────────────────────────────────────────────────
def hagrid_bbox_to_xyxy(bbox):
    """HaGRID [x1, y1, w, h] normalised → [x1, y1, x2, y2] normalised."""
    x1, y1, w, h = bbox  # top-left format
    return [x1, y1, x1 + w, y1 + h]


def load_gt(dataset_dir: Path) -> dict:
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
            boxes_xyxy = []
            for b in entry.get("bboxes", []):
                x1, y1, x2, y2 = hagrid_bbox_to_xyxy(b)
                # clamp to image bounds
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(1.0, x2)
                y2 = min(1.0, y2)
                boxes_xyxy.append([x1, y1, x2, y2])
            gt_map[uuid] = {"class_idx": cls_idx, "boxes": boxes_xyxy}
    print(f"  Loaded GT for {len(gt_map):,} images")
    return gt_map


def uuid_from_filename(stem: str) -> str:
    return stem.split("_jpg.rf.")[0] if "_jpg.rf." in stem else stem


# ── IoU & mAP ─────────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0]);  yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]);  yB = min(box_a[3], box_b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_allpoint(recalls, precisions):
    """All-point interpolation AP — VOC 2010+ standard."""
    recalls    = np.concatenate(([0.0], recalls,    [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def compute_map(all_results: list) -> dict:
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

            # Detections for this class: score is col cls_idx (1=palm, 2=two)
            scores = res["pred_scores"][:, cls_idx]   # shape (N,)
            boxes  = res["pred_boxes"]                # shape (N, 4)

            matched_gt = set()
            order      = np.argsort(-scores)

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
            aps[cls_name] = float("nan");  continue
        if not detections:
            aps[cls_name] = 0.0;           continue

        detections.sort(key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in detections]).astype(float)
        fp_cum = np.cumsum([1 - d[1] for d in detections]).astype(float)
        recalls    = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        aps[cls_name] = compute_ap_allpoint(recalls, precisions)

    return aps


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(dataset_dir: str, model_path: str):
    dataset_dir = Path(dataset_dir)

    print("\n[1/3] Loading ground truth …")
    gt_map = load_gt(dataset_dir)
    # after gt_map = load_gt(dataset_dir) added temporary
    print("Sample GT keys:", list(gt_map.keys())[:3])
        
    sample_uuid = list(gt_map.keys())[0]
    print("Sample GT entry:", gt_map[sample_uuid])
    #added temporary

    print("\n[2/3] Loading TFLite model …")
    interp, inp_details, out_details = load_interpreter(model_path)
    print(f"  Input  shape: {inp_details['shape']}  dtype: {inp_details['dtype']}")
    print(f"  Output shape: {out_details['shape']}  dtype: {out_details['dtype']}")

    print("\n[3/3] Running inference …\n")
    test_dir    = dataset_dir / "test"
    all_results = []
    correct     = defaultdict(int)
    total       = defaultdict(int)
    conf_matrix = np.zeros((2, 3), dtype=int)  # rows=GT, cols=pred+no_detect

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
            uuid     = uuid_from_filename(img_path.stem)
            # --- temporary diagnostic --- added
            if total[gesture] < 5:
                print(f"stem: {img_path.stem} → uuid: {uuid} → in GT: {uuid in gt_map}")
            # ---temp
            gt_entry = gt_map.get(uuid, {})
            gt_boxes = gt_entry.get("boxes", [])

            try:
                img_q = preprocess(img_path)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            # Shape: (172, 7) → [bg, palm, two, left, top, right, bottom]
            result      = run_inference(interp, inp_details, out_details, img_q)

            # --- temporary diagnostic ---
            if total[gesture] == 0:
                scores = result[:, 1]
                best = np.argmax(scores)
                print(f"Best pred box: {result[best, 3:]}")
                print(f"GT boxes for this image: {gt_boxes}")
                print(f"Best pred score: {scores[best]:.3f}")
                # score distribution
                all_scores = result[:, 1]
                print(f"Score distribution: min={all_scores.min():.3f} "
                      f"max={all_scores.max():.3f} "
                      f"mean={all_scores.mean():.3f} "
                      f">0.5: {(all_scores>0.5).sum()} "
                      f">0.1: {(all_scores>0.1).sum()} "
                      f">0.01: {(all_scores>0.01).sum()}")
            # ----------------------------
            pred_scores = result[:, :3]   # (172, 3)  — bg, palm, two
            pred_boxes  = result[:, 3:]   # (172, 4)  — x1, y1, x2, y2

            total[gesture] += 1

            # ── Per-image accuracy (ACC_THRESHOLD) ───────────────────────
            # Best gesture score per detection (ignore background col 0)
            best_cls_scores = pred_scores[:, 1:].max(axis=1)   # (172,)
            best_cls_ids    = pred_scores[:, 1:].argmax(axis=1) + 1  # 1=palm,2=two

            high_conf = best_cls_scores >= ACC_THRESHOLD
            if not high_conf.any():
                conf_matrix[gt_idx, 2] += 1   # no detection
            else:
                # Take highest-confidence detection overall
                hc_scores = best_cls_scores[high_conf]
                hc_ids    = best_cls_ids[high_conf]
                best      = np.argmax(hc_scores)
                pred_cls  = int(hc_ids[best])
                pred_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else "unknown"

                if pred_cls == gt_class_idx:
                    correct[gesture] += 1
                    conf_matrix[gt_idx, gt_idx] += 1
                else:
                    pred_row = GESTURE_NAMES.index(pred_name) \
                               if pred_name in GESTURE_NAMES else 2
                    conf_matrix[gt_idx, pred_row] += 1

            # ── Collect for mAP (all detections) ─────────────────────────
            # ── Apply NMS per class before collecting for mAP ─────────────
            import cv2
            nms_scores_all = []
            nms_boxes_all  = []

            for cls_idx in range(1, 3):  # 1=palm, 2=two
                cls_scores = pred_scores[:, cls_idx]
                above = cls_scores >= MAP_THRESHOLD
                if not above.any():
                    continue

                boxes_above  = pred_boxes[above]   # (N, 4) x1y1x2y2
                scores_above = cls_scores[above]   # (N,)

                # cv2.dnn.NMSBoxes expects [x,y,w,h] and score as list
                boxes_xywh = []
                for b in boxes_above:
                    x1, y1, x2, y2 = b
                    boxes_xywh.append([float(x1), float(y1),
                                       float(x2-x1), float(y2-y1)])

                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh,
                    scores_above.tolist(),
                    score_threshold=float(MAP_THRESHOLD),
                    nms_threshold=0.3
                )

                if len(indices) == 0:
                    continue

                indices = indices.flatten()
                for i in indices:
                    nms_scores_all.append((cls_idx, float(scores_above[i])))
                    nms_boxes_all.append(boxes_above[i])

            # Build filtered pred arrays for mAP
            if nms_scores_all:
                n = len(nms_scores_all)
                filtered_scores = np.zeros((n, 3), dtype=np.float32)
                filtered_boxes  = np.zeros((n, 4), dtype=np.float32)
                for i, (cls_idx, score) in enumerate(nms_scores_all):
                    filtered_scores[i, cls_idx] = score
                    filtered_boxes[i] = nms_boxes_all[i]
            else:
                filtered_scores = np.zeros((1, 3), dtype=np.float32)
                filtered_boxes  = np.zeros((1, 4), dtype=np.float32)

            all_results.append({
                "gt_class_idx": gt_class_idx,
                "gt_boxes":     gt_boxes,
                "pred_scores":  filtered_scores,
                "pred_boxes":   filtered_boxes,
            })

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  RESULTS  —  INT8 TFLite")
    print("═" * 58)

    print(f"\n── Per-class Detection Accuracy  (conf ≥ {ACC_THRESHOLD:.2f}) ──────")
    overall_correct = overall_total = 0
    for gesture in GESTURE_NAMES:
        if total[gesture] == 0:
            print(f"  {gesture:>6}: no images found");  continue
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
    parser.add_argument("--dataset_dir",    type=str,   required=True)
    parser.add_argument("--model_path",     type=str,   required=True)
    parser.add_argument("--acc_threshold",  type=float, default=0.5,
                        help="Confidence threshold for accuracy/confusion matrix (default: 0.5)")
    parser.add_argument("--iou_threshold",  type=float, default=0.5,
                        help="IoU threshold for mAP TP/FP matching (default: 0.5)")
    args = parser.parse_args()

    ACC_THRESHOLD = args.acc_threshold
    IOU_THRESHOLD = args.iou_threshold
    evaluate(args.dataset_dir, args.model_path)
