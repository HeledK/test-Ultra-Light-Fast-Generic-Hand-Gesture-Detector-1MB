import tensorflow as tf
import numpy as np
import cv2

interp = tf.lite.Interpreter(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\tflite files\FYP2 multiclass 2000 images\ultraslim_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

img = cv2.imread(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\data\wider_face_add_lm_10_10\dataset\test\two\00a05de0-b5dd-4242-9d30-7e26604a7ac4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128)).astype(np.float32)

img_q = np.clip(np.round(img / 0.0078125 + (-1)), -128, 127).astype(np.int8)
img_q = img_q[np.newaxis]

interp.set_tensor(inp['index'], img_q)
interp.invoke()

raw    = interp.get_tensor(out['index'])
result = (raw.astype(np.float32) - (-128)) * 0.003921568859368563

print("All 7 column ranges on a real TWO image:")
for i in range(7):
    print(f"  col {i}: min={result[:,i].min():.4f}  max={result[:,i].max():.4f}  mean={result[:,i].mean():.4f}")

print("\nFull unique values in each column:")
for i in range(7):
    unique = np.unique(np.round(result[:,i], 3))
    print(f"  col {i}: {unique[:10]}")  # first 10 unique values