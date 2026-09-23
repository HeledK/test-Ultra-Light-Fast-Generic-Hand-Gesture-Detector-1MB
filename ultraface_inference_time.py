import time
import tensorflow as tf
import numpy as np
interp = tf.lite.Interpreter("models/ultraslim_int8.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

dummy = np.zeros(inp['shape'], dtype=np.int8)
# warmup
for _ in range(10):
    interp.set_tensor(inp['index'], dummy)
    interp.invoke()
# measure
times = []
for _ in range(100):
    t = time.perf_counter()
    interp.set_tensor(inp['index'], dummy)
    interp.invoke()
    times.append((time.perf_counter() - t) * 1000)
print(f"Avg: {sum(times)/len(times):.2f}ms  Min: {min(times):.2f}ms")