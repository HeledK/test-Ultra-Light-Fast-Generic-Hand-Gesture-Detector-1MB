import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

MODEL_PATH  = "models/ultraslim_int8.tflite"
DATASET_DIR = Path(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\data\wider_face_add_lm_10_10\dataset")
N_IMAGES    = 20  # just check 20 images

INP_SCALE      = 0.0078125
INP_ZERO_POINT = -1
OUT_SCALE      = 0.003921568859368563
OUT_ZERO_POINT = -128

def hagrid_bbox_to_xyxy(bbox):
    cx, cy, w, h = bbox
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def compute_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# load model
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

# load a few GT boxes
with open(DATASET_DIR / "palm_test.json") as f:
    data = json.load(f)

image_dir = DATASET_DIR / "test" / "palm"
count = 0
ious = []

for uuid, entry in list(data.items())[:N_IMAGES]:
    img_path = image_dir / f"{uuid}.jpg"
    if not img_path.exists():
        continue

    # preprocess
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128)).astype(np.float32)
    img = (img - 127.0) / 128.0
    img_q = np.clip(np.round(img / INP_SCALE + INP_ZERO_POINT), -128, 127).astype(np.int8)
    img_q = img_q[np.newaxis]

    # infer
    interp.set_tensor(inp['index'], img_q)
    interp.invoke()
    raw = interp.get_tensor(out['index'])
    result = (raw.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE

    pred_scores = result[:, 1]  # palm scores
    pred_boxes  = result[:, 3:]

    # best pred box
    best_idx = np.argmax(pred_scores)
    best_score = pred_scores[best_idx]
    best_box = pred_boxes[best_idx]

    # GT boxes (clamped)
    gt_boxes = []
    for b in entry.get("bboxes", []):
        x1, y1, x2, y2 = hagrid_bbox_to_xyxy(b)
        gt_boxes.append([max(0,x1), max(0,y1), min(1,x2), min(1,y2)])

    # best IoU against any GT box
    best_iou = max((compute_iou(best_box, gt) for gt in gt_boxes), default=0.0)
    ious.append(best_iou)

    print(f"{uuid[:8]}  score={best_score:.3f}  pred={np.round(best_box,3)}  "
          f"gt={np.round(gt_boxes[0],3) if gt_boxes else 'none'}  IoU={best_iou:.3f}")
    count += 1

print(f"\nMean IoU over {count} images: {np.mean(ious):.3f}")
print(f"IoU >= 0.5: {sum(i >= 0.5 for i in ious)} / {count}")