"""
compare_boxes.py
Runs both pytorch and tflite on the same 20 palm images and prints
pred box, GT box, and IoU side by side for each.
"""
import json
import numpy as np
import cv2
import torch
import tensorflow as tf
from pathlib import Path
from torch.nn import Conv2d, Sequential, ModuleList, ReLU

# ── Paths — edit these ────────────────────────────────────────────────────────
PTH_PATH     = r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\train-version-slim\slim-Epoch-19-Loss-1.3659555289149283\slim-Epoch-19-Loss-1.3659555289149283.pth"
TFLITE_PATH  = "models/ultraslim_int8.tflite"
DATASET_DIR  = Path(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\data\wider_face_add_lm_10_10\dataset")
N_IMAGES     = 20
# ─────────────────────────────────────────────────────────────────────────────

INP_SCALE      = 0.0078125
INP_ZERO_POINT = -1
OUT_SCALE      = 0.003921568859368563
OUT_ZERO_POINT = -128

def hagrid_bbox_to_xyxy(bbox):
    cx, cy, w, h = bbox
    return [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]

def compute_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

# ── TFLite setup ──────────────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

def tflite_predict(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128)).astype(np.float32)
    img = (img - 127.0) / 128.0
    img_q = np.clip(np.round(img / INP_SCALE + INP_ZERO_POINT), -128, 127).astype(np.int8)
    interp.set_tensor(inp_det['index'], img_q[np.newaxis])
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])
    result = (raw.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE
    best = np.argmax(result[:, 1])  # best palm score
    return result[best, 1], result[best, 3:]  # score, box

# ── PyTorch setup ─────────────────────────────────────────────────────────────
from vision.ssd.config import fd_config as config
from vision.ssd.ssd import SSD
from vision.nn.mb_tiny import Mb_Tiny
from vision.ssd.predictor import Predictor

def build_model(num_classes, device):
    base_net = Mb_Tiny(2)
    ch = base_net.base_channel
    extras = ModuleList([Sequential(
        Conv2d(ch*16, ch*4, kernel_size=1), ReLU(),
        Sequential(Conv2d(ch*4, ch*4, 3, 2, 1, groups=ch*4), ReLU(),
                   Conv2d(ch*4, ch*16, 1)), ReLU()
    )])
    regression_headers = ModuleList([
        Sequential(Conv2d(ch*4,  ch*4,  3,1,1,groups=ch*4), ReLU(), Conv2d(ch*4,  3*4, 1)),
        Sequential(Conv2d(ch*8,  ch*8,  3,1,1,groups=ch*8), ReLU(), Conv2d(ch*8,  2*4, 1)),
        Sequential(Conv2d(ch*16, ch*16, 3,1,1,groups=ch*16),ReLU(), Conv2d(ch*16, 2*4, 1)),
        Conv2d(ch*16, 3*4, 3, 1, 1),
    ])
    classification_headers = ModuleList([
        Sequential(Conv2d(ch*4,  ch*4,  3,1,1,groups=ch*4), ReLU(), Conv2d(ch*4,  3*num_classes,1)),
        Sequential(Conv2d(ch*8,  ch*8,  3,1,1,groups=ch*8), ReLU(), Conv2d(ch*8,  2*num_classes,1)),
        Sequential(Conv2d(ch*16, ch*16, 3,1,1,groups=ch*16),ReLU(), Conv2d(ch*16, 2*num_classes,1)),
        Conv2d(ch*16, 3*num_classes, 3, 1, 1),
    ])
    return SSD(num_classes, base_net.model, [8,11,13],
               extras, classification_headers, regression_headers,
               is_test=True, config=config, device=device)

config.define_img_size(320)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = build_model(3, device)
net.load(PTH_PATH)
net.to(device)
net.eval()

predictor = Predictor(net, config.image_size, config.image_mean_test,
                      config.image_std, nms_method=None,
                      iou_threshold=config.iou_threshold,
                      candidate_size=200, sigma=0.5, device=device)

def pth_predict(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (320, 240))
    with torch.no_grad():
        boxes, labels, scores = predictor.predict(img_resized, top_k=1, prob_threshold=0.01)
    if len(scores) == 0:
        return 0.0, None
    boxes  = boxes.numpy()  if torch.is_tensor(boxes)  else np.array(boxes)
    scores = scores.numpy() if torch.is_tensor(scores) else np.array(scores)
    # normalise to [0,1]
    boxes = boxes / np.array([320, 240, 320, 240], dtype=float)
    best = np.argmax(scores)
    return float(scores[best]), boxes[best]

# ── Run comparison ────────────────────────────────────────────────────────────
with open(DATASET_DIR / "palm_test.json") as f:
    data = json.load(f)

image_dir = DATASET_DIR / "test" / "palm"

print(f"\n{'uuid':>10}  {'pth_score':>9}  {'pth_iou':>7}  {'tfl_score':>9}  {'tfl_iou':>7}  pth_box   tfl_box   gt_box")
print("-" * 120)

count = 0
pth_ious = []
tfl_ious = []

for uuid, entry in list(data.items())[:N_IMAGES]:
    img_path = image_dir / f"{uuid}.jpg"
    if not img_path.exists():
        img_path = image_dir / f"{uuid}.png"
    if not img_path.exists():
        continue

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    gt_boxes = [hagrid_bbox_to_xyxy(b) for b in entry.get("bboxes", [])]
    if not gt_boxes:
        continue
    gt = gt_boxes[0]

    pth_score, pth_box = pth_predict(img_bgr)
    tfl_score, tfl_box = tflite_predict(img_bgr)

    pth_iou = compute_iou(pth_box, gt) if pth_box is not None else 0.0
    tfl_iou = compute_iou(tfl_box, gt) if tfl_box is not None else 0.0

    pth_ious.append(pth_iou)
    tfl_ious.append(tfl_iou)

    pth_b = np.round(pth_box, 3) if pth_box is not None else [None]*4
    tfl_b = np.round(tfl_box, 3)
    gt_r  = np.round(gt, 3)

    print(f"{uuid[:8]}  {pth_score:>9.3f}  {pth_iou:>7.3f}  "
          f"{tfl_score:>9.3f}  {tfl_iou:>7.3f}  "
          f"{pth_b}  {tfl_b}  {gt_r}")
    count += 1

print("-" * 120)
print(f"Mean IoU  —  PyTorch: {np.mean(pth_ious):.3f}   TFLite: {np.mean(tfl_ious):.3f}  (over {count} images)")