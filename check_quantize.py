# Run this first before editing:
import tensorflow as tf
interp = tf.lite.Interpreter("models/ultraslim_int8.tflite")
interp.allocate_tensors()
print(interp.get_input_details()[0]['quantization'])   # gives (scale, zero_point)
print(interp.get_output_details()[0]['quantization'])  # gives (scale, zero_point)