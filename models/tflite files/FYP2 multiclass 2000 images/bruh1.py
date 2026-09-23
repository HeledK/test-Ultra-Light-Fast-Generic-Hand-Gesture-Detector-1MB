import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\tflite files\FYP2 multiclass 2000 images\ultraslim_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

# Dequantize output using scale and zero_point from get_output_details
scale      = 0.003921568859368563
zero_point = -128

dummy = np.zeros((1, 128, 128, 3), dtype=np.int8)
interp.set_tensor(inp['index'], dummy)
interp.invoke()

raw    = interp.get_tensor(out['index'])          # INT8
result = (raw.astype(np.float32) - zero_point) * scale   # float

print("Output shape:", result.shape)
print("First 5 rows (dequantized):")
print(np.round(result[:5], 4))
print("\nValue ranges per column:")
for i in range(7):
    print(f"  col {i}: min={result[:,i].min():.4f}  max={result[:,i].max():.4f}")