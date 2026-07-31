#!/usr/bin/env bash
set -e
set -o pipefail

cd "$(dirname "$0")/../.."

mkdir -p scale_cache/vtpq logs

export PYTHONPATH="$PWD/3rdparty/lmms-eval:$PWD/3rdparty/LLaVA-NeXT:$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
DATA="data/coco_calib_512.json"
IMAGE_FOLDER="data/sharegpt4v"
SCALE="scale_cache/vtpq/llava_onevision_7b_w4a8_keep090_lam050.pt"

CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_quant.py \
  --model llava_onevision \
  --model_args "pretrained=$MODEL" \
  --calib_data coco \
  --data_path "$DATA" \
  --image_folder "$IMAGE_FOLDER" \
  --n_samples 128 \
  --micro_batch_size 1 \
  --method vtpq \
  --run_process \
  --w_bit 4 \
  --a_bit 8 \
  --w_group 128 \
  --reweight \
  --distort \
  --loss_mode mae \
  --vtpq_keep_ratio 0.90 \
  --vtpq_lambda 0.50 \
  --vtpq_score_bit 8 \
  --vtpq_always_keep_front 0 \
  --vtpq_score_source embed \
  --vtpq_score_layers 4 \
  --vtpq_positive_only \
  --vtpq_min_prune_score 0.0 \
  --vtpq_scale_loss robust \
  --vtpq_tail_ratio 0.20 \
  --vtpq_tail_weight 0.25 \
  --scale_path "$SCALE" \
  2>&1 | tee logs/vtpq_llava_onevision_7b_w4a8_keep090_lam050.log
