import numpy as np
import tensorflow as tf
import cv2

MODEL_PATH = "models/ultraslim_int8.tflite"
IMAGE_PATH = "0b2f8b45-d818-4055-8d98-63fd500e88ce.jpg"
THRESHOLD = 0.3
INPUT_SIZE = (128, 128)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale = input_details[0]['quantization_parameters']['scales'][0]
in_zp    = input_details[0]['quantization_parameters']['zero_points'][0]
out_scale = output_details[0]['quantization_parameters']['scales'][0]
out_zp    = output_details[0]['quantization_parameters']['zero_points'][0]

# Load and preprocess
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, INPUT_SIZE)

image_float = (image_resized.astype(np.float32) - 127.0) / 128.0
image_int8 = np.clip(np.round(image_float / in_scale + in_zp), -128, 127).astype(np.int8)
image_int8 = image_int8[np.newaxis, ...]  # shape: [1, 128, 128, 3]

# Run inference
interpreter.set_tensor(input_details[0]['index'], image_int8)
interpreter.invoke()

# Dequantize output
raw_output = interpreter.get_tensor(output_details[0]['index'])  # shape [172, 7] or [1, 172, 7]
output = (raw_output.astype(np.float32) - out_zp) * out_scale

# Handle missing batch dim
if output.ndim == 2:
    output = output[np.newaxis, ...]  # force to [1, 172, 7]

detections = output[0]  # shape [172, 7]

print(f"Total anchors: {len(detections)}")
print(f"Max palm score:  {detections[:, 1].max():.4f}")
print(f"Max two score:   {detections[:, 2].max():.4f}")
print(f"Max bg score:    {detections[:, 0].max():.4f}")

# Filter detections above threshold
for i, det in enumerate(detections):
    bg_score, palm_score, two_score = det[0], det[1], det[2]
    if palm_score > THRESHOLD or two_score > THRESHOLD:
        label = "palm" if palm_score > two_score else "two"
        confidence = max(palm_score, two_score)
        print(f"Anchor {i}: {label} ({confidence:.3f}) | box: {det[3:]}")