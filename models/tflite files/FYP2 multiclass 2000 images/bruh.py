import tensorflow as tf
interp = tf.lite.Interpreter(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\tflite files\FYP2 multiclass 2000 images\ultraslim_int8.tflite")
interp.allocate_tensors()
print("INPUTS:", interp.get_input_details())
print("OUTPUTS:", interp.get_output_details())

