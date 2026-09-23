"""
This code is to produce ultraslim version.
"""
import cv2, numpy as np, tensorflow as tf, glob, random
import os
import argparse
import tensorflow as tsf

random.seed(1337)

SAVE_MODEL_DIR = 'tf/export_models/slim_ultraslim/'
OUTPUT_TF_FILE_NAME_INT8 = 'models/ultraslim_int8.tflite'

parser = argparse.ArgumentParser(
    description='train With Pytorch')
parser.add_argument('--checkpoint_folder', default='data/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

IMAGES_PATH = os.path.join(args.checkpoint_folder, "wider_face_add_lm_10_10\JPEGImages\*jpg" )  #changed this to png from jpg
NIMAGES_REPRESENTATIVE_DATASET = 400  #was 100                                                            #THEN CHANGED BACK TO jpg

def representative_dataset():
    files = glob.glob(os.path.join(args.checkpoint_folder, "wider_face_add_lm_10_10/JPEGImages/*.jpg"))#added these two replaced another
    #files += glob.glob(os.path.join(args.checkpoint_folder, "wider_face_add_lm_10_10/JPEGImages/*.png"))

    print("Representative dataset images:", len(files)) #added
    random.shuffle(files)

    for filename in files[:NIMAGES_REPRESENTATIVE_DATASET]:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # added rgb conversion
        image = cv2.resize(image, (320, 240))
        image = (image[None, ...] - 127.0) / 128.0
        yield [image.astype(np.float32)]

IMAGE_FILENAME = "example_input.jpg"
BOX_COLOR = (255, 128, 0)
THRESHOLD = 0.5

def main():
  
    converter = tsf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    converter.optimizations = [tsf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tsf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tsf.int8
    converter.inference_output_type = tsf.int8

    tflite_model = converter.convert()
    open(OUTPUT_TF_FILE_NAME_INT8, 'wb').write(tflite_model)
    
    # Inspect the converted model
    interpreter = tsf.lite.Interpreter(model_path=OUTPUT_TF_FILE_NAME_INT8)

    interpreter.allocate_tensors()

    # Print input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    # Check the data type of the weights
    for tensor in interpreter.get_tensor_details():
        print(tensor['name'], tensor['dtype'])

 
if __name__ == '__main__':
    main()