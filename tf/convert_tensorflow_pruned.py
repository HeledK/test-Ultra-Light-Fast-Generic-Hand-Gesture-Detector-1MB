"""
convert_tensorflow_pruned.py
─────────────────────────────
Convert a pruned PyTorch model (from step3 fine-tuning) to TensorFlow
for subsequent INT8 TFLite quantization via ultraslim.py.

Usage:
    python convert_tensorflow_pruned.py \
        --torch_path ../models/pruning/finetuned_best.pth \
        --pruned_widths "12,24,24,24,52,56,60,64,112,88,128,248,256" \
        --output_dir export_models/slim_pruned_ultraslim/

Prerequisites:
    - finetuned_best.pth from step3 (PyTorch)
    - mapping_tables/slim_320.json (same as original, layer names didn't change)
    - model/slim_320_pruned.py (the new pruned TF model builder)
"""

import argparse
import sys

from backend.utils_test import load_weight
from model.slim_320_pruned import create_slim_net_pruned

parser = argparse.ArgumentParser(description='Convert pruned PyTorch model to TF')
parser.add_argument('--torch_path',    required=True,
                    help='Path to pruned/finetuned .pth file')
parser.add_argument('--pruned_widths', required=True,
                    help='Comma-separated 13 ints, e.g. "12,24,24,24,52,56,60,64,112,88,128,248,256"')
parser.add_argument('--mapping_table', default='mapping_tables/slim_320.json',
                    help='JSON mapping table (same as for original model)')
parser.add_argument('--output_dir',    default='export_models/slim_pruned_ultraslim/',
                    help='Where to save the TF model')
parser.add_argument('--input_size',    default=128, type=int,
                    help='Deployment input size (square)')
args = parser.parse_args()


def main():             #default args.input_size,args.input_size
    input_shape = (args.input_size,args.input_size)   #temporarily put it to 240, 320 H,W
    num_classes = 3             # H,W  if want to modify the thing you have to go into slim_320.py and change the thing
                                #also change import from utils to utils_test
    widths = [int(x.strip()) for x in args.pruned_widths.split(',')]
    if len(widths) != 13:
        print(f'ERROR: expected 13 widths, got {len(widths)}')
        sys.exit(1)

    print(f'Pruned widths: {widths}')
    print(f'Loading PyTorch weights from: {args.torch_path}')
    print(f'Building pruned TF model at input {input_shape} ...')

    model = create_slim_net_pruned(input_shape, widths, num_classes)

    print(f'Loading weights via mapping table: {args.mapping_table}')
    load_weight(model, args.torch_path, args.mapping_table)

    print(f'Saving TF model to: {args.output_dir}')
    model.save(args.output_dir, include_optimizer=False)

    print('Done.')
    print(f'Next: edit ultraslim.py SAVE_MODEL_DIR to "{args.output_dir}" and run it.')


if __name__ == '__main__':
    main()
