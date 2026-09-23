import cv2
import numpy as np
import tensorflow as tf
import glob
import os
import random
import argparse

random.seed(1337)

SAVE_MODEL_DIR = 'tf/export_models/slim_ultraslim/'
OUTPUT_TF_FILE_NAME_INT8 = 'models/ultraslim_int8.tflite'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_folder', default='data/')
args = parser.parse_args()

NIMAGES_REPRESENTATIVE_DATASET = 400  # was 100

# MUST match training input size
INPUT_W, INPUT_H = 128, 128

# MUST match training color order — verify against HaGRIDDataset
USE_RGB = True  # flip to False if training uses BGR

def representative_dataset():
    patterns = [
        os.path.join(args.checkpoint_folder, "wider_face_add_lm_10_10/JPEGImages/*.jpg"),
        os.path.join(args.checkpoint_folder, "wider_face_add_lm_10_10/JPEGImages/*.png"),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p)
    print(f"Representative dataset pool: {len(files)} images")

    random.shuffle(files)
    for filename in files[:NIMAGES_REPRESENTATIVE_DATASET]:
        image = cv2.imread(filename)
        if image is None:
            continue
        if USE_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_W, INPUT_H))
        image = (image[None, ...].astype(np.float32) - 127.0) / 128.0
        yield [image]

def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(OUTPUT_TF_FILE_NAME_INT8), exist_ok=True)
    with open(OUTPUT_TF_FILE_NAME_INT8, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=OUTPUT_TF_FILE_NAME_INT8)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input details:", input_details)
    print("Output details:", output_details)
    print(f"Input quantization (scale, zero_point): {input_details[0]['quantization']}")

if __name__ == '__main__':
    main()