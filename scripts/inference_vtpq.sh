#!/usr/bin/env bash
set -e
set -o pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs eval_logs

export PYTHONPATH="$PWD/3rdparty/lmms-eval:$PWD/3rdparty/LLaVA-NeXT:$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL="OpenGVLab/InternVL2-8B"
SCALE="scale_cache/vtpq/internvl2_w4a8_keep090_lam050.pt"
INPUT="inference/question.json"
OUTPUT="outputs/internvl2_vtpq_keep090.json"
LOG="eval_logs/internvl2_vtpq_keep090.log"

CUDA_VISIBLE_DEVICES=0 python3 -W ignore inference.py \
  --model internvl2 \
  --model_args "pretrained=$MODEL" \
  --device cuda \
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
  --infer_pairs "$INPUT" \
  --save_path "$OUTPUT" \
  --max_new_tokens 128 \
  2>&1 | tee "$LOG"
