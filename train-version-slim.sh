#!/usr/bin/env bash
model_root_path="./models/train-version-slim"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python -u train.py \
  --datasets \
  ./data/wider_face_add_lm_10_10 \
  --validation_dataset \
  ./data/wider_face_add_lm_10_10 \
  --net \
  slim \
  --num_epochs \
  39 \
  --milestones \
  "19,34" \
  --lr \
  5e-4 \
  --batch_size \
  32 \
  --input_size \
  320 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  6 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  0 \
  --resume \
  models/train-version-slim/slim-Epoch-20-Loss-4.218339783080081/slim-Epoch-20-Loss-4.218339783080081.pth \
  2>&1 | tee "$log"
