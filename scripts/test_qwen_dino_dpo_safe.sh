#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="data/iu_xray/images"
version="qwen_dino_dpo_safe_ref_cap50_margin010_test"
savepath="./save/${dataset}/${version}"
default_delta=$(ls -t save/iu_xray/qwen_dino_dpo_safe_ref_cap50_margin010/checkpoints/*_train.pth 2>/dev/null | head -1)
delta_file="${DELTA_FILE:-${default_delta}}"

if [ -z "${delta_file}" ]; then
  echo "No safe DPO checkpoint found. Set DELTA_FILE or run qwen_dino_dpo_iuxray_8gpu_safe.sh first." >&2
  exit 1
fi

mkdir -p "${savepath}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --vision_model resources/models/facebook_dinov2-base \
    --llm_model resources/models/Qwen3-8B \
    --llm_use_lora True \
    --llm_r 16 \
    --llm_alpha 32 \
    --resampler_num_queries 32 \
    --resampler_num_layers 2 \
    --test_batch_size 16 \
    --max_length 160 \
    --min_new_tokens 40 \
    --max_new_tokens 160 \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 1 \
    --strategy auto \
    --accelerator gpu \
    --precision bf16-mixed \
    2>&1 | tee ${savepath}/log.txt
