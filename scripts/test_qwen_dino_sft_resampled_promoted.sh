#!/bin/bash
set -euo pipefail

cd /data/wang.xuanmin/R2GenGPT

delta_file="save/iu_xray/qwen_dino_sft_resampled_abnormal_lr1e6/checkpoints/checkpoint_epoch0_step175_train.pth"
savepath="save/iu_xray/qwen_dino_sft_resampled_abnormal_lr1e6_test_promoted"

mkdir -p "${savepath}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train.py \
  --test \
  --dataset iu_xray \
  --annotation data/iu_xray/annotation.json \
  --base_dir data/iu_xray/images \
  --delta_file "${delta_file}" \
  --vision_model resources/models/facebook_dinov2-base \
  --llm_model resources/models/Qwen3-8B \
  --llm_use_lora True \
  --llm_r 16 \
  --llm_alpha 32 \
  --resampler_num_queries 32 \
  --resampler_num_layers 2 \
  --test_batch_size 16 \
  --max_length 160 \
  --beam_size 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 2 \
  --min_new_tokens 40 \
  --max_new_tokens 120 \
  --freeze_vm True \
  --savedmodel_path "${savepath}" \
  --num_workers 8 \
  --devices 1 \
  --strategy auto \
  --accelerator gpu \
  --precision bf16-mixed \
  2>&1 | tee "${savepath}/run.log"
