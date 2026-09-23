import tensorflow as tf
import numpy as np
import cv2

from model.slim_320 import create_slim_net
from backend.utils import load_weight

input_shape = (128, 128)
base_channel = 16
num_classes = 3

m = create_slim_net(input_shape, base_channel, num_classes)
load_weight(m,
            r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\train-version-slim\slim-Epoch-19-Loss-1.3659555289149283\slim-Epoch-19-Loss-1.3659555289149283.pth",
            r"mapping_tables\slim_320.json")

img = cv2.imread(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\data\wider_face_add_lm_10_10\dataset\test\two\00a05de0-b5dd-4242-9d30-7e26604a7ac4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128)).astype(np.float32)
img = (img - 127.0) / 128.0
img = img[np.newaxis]

# ← key difference: training=False forces BatchNorm to use running stats
result = m(img, training=False).numpy()

best = result[:, 1:3].max(axis=1)
top5 = np.argsort(-best)[:5]
print("Top 5 detections (training=False):")
print("  [bg, palm, two, x1, y1, x2, y2]")
for i in top5:
    print(f"  {np.round(result[i], 4).tolist()}")

print(f"\nMax palm: {result[:,1].max():.4f}")
print(f"Max two:  {result[:,2].max():.4f}")
print(f"Max bg:   {result[:,0].max():.4f}")