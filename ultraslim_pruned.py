"""
ultraslim_pruned.py
───────────────────
INT8 TFLite quantization of the pruned TF model.
Identical to ultraslim.py except for input/output paths.
"""
import cv2, numpy as np, tensorflow as tf, glob, random
import os
import argparse
import tensorflow as tsf

random.seed(1337)

SAVE_MODEL_DIR = 'tf/export_models/slim_pruned_ultraslim/'
OUTPUT_TF_FILE_NAME_INT8 = 'models/ultraslim_pruned_int8.tflite'

parser = argparse.ArgumentParser(description='INT8 quantize pruned TF model')
parser.add_argument('--checkpoint_folder', default='data/',
                    help='Directory containing wider_face_add_lm_10_10/JPEGImages/')
parser.add_argument('--save_model_dir', default=SAVE_MODEL_DIR,
                    help='Path to the saved TF model from convert_tensorflow_pruned.py')
parser.add_argument('--output_path', default=OUTPUT_TF_FILE_NAME_INT8,
                    help='Where to write the .tflite file')
args = parser.parse_args()

NIMAGES_REPRESENTATIVE_DATASET = 400


def representative_dataset():
    files = glob.glob(os.path.join(args.checkpoint_folder,
                                   "wider_face_add_lm_10_10/JPEGImages/*.jpg"))
    print("Representative dataset images:", len(files))
    random.shuffle(files)

    for filename in files[:NIMAGES_REPRESENTATIVE_DATASET]:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240)) #was 128,128 | added
        image = (image[None, ...] - 127.0) / 128.0
        yield [image.astype(np.float32)]


def main():
    print(f'Loading TF model from: {args.save_model_dir}')
    converter = tsf.lite.TFLiteConverter.from_saved_model(args.save_model_dir)
    converter.optimizations = [tsf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tsf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tsf.int8
    converter.inference_output_type = tsf.int8

    print('Converting and quantizing ...')
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Wrote: {args.output_path}')

    file_size_kb = os.path.getsize(args.output_path) / 1024
    print(f'File size: {file_size_kb:.1f} KB')

    # Inspect the converted model
    interpreter = tsf.lite.Interpreter(model_path=args.output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)


if __name__ == '__main__':
    main()
