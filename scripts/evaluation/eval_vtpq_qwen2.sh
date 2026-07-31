#!/usr/bin/env bash
set -e
set -o pipefail

cd "$(dirname "$0")/../.."

mkdir -p outputs/qwen2_vl_7b_w4a8_keep090_pruned

export PYTHONPATH="$PWD/3rdparty/lmms-eval:$PWD/3rdparty/LLaVA-NeXT:$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL="Qwen/Qwen2-VL-7B-Instruct"
SCALE="scale_cache/vtpq/qwen2_vl_7b_w4a8_keep090_lam050.pt"
OUT_DIR="outputs/qwen2_vl_7b_w4a8_keep090_pruned"
TASKS="scienceqa_img,vizwiz_vqa_val,chartqa,ai2d,mmmu_val,ocrbench"

CUDA_VISIBLE_DEVICES=0 python3 -W ignore main.py \
  --model qwen2_vl \
  --model_args "pretrained=$MODEL" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --method vtpq \
  --pseudo_quant \
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
  --vtpq_infer_prune \
  --vtpq_infer_positive_only \
  --vtpq_infer_min_score 0.0 \
  --scale_path "$SCALE" \
  --log_samples \
  --log_samples_suffix "vtpq_qwen2_keep090" \
  --output_path "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/eval.log"
