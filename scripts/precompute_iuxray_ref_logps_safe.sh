#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="data/iu_xray/images"
input_preferences="data/iu_xray/preferences_cap50_margin010.jsonl"
output_preferences="data/iu_xray/preferences_cap50_margin010_ref.jsonl"
sft_delta_file="save/iu_xray/qwen_dino_sft_8gpu_continue2/checkpoints/checkpoint_epoch1_step190_train.pth"
log="save/iu_xray/qwen_dino_dpo_safe_ref_cap50_margin010/precompute_ref_logps.log"

mkdir -p "$(dirname "${log}")"

CUDA_VISIBLE_DEVICES=0 python -u precompute_reference_logps.py \
    --stage dpo \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --preference_file ${input_preferences} \
    --output_preference_file ${output_preferences} \
    --delta_file ${sft_delta_file} \
    --vision_model resources/models/facebook_dinov2-base \
    --llm_model resources/models/Qwen3-8B \
    --llm_use_lora True \
    --llm_r 16 \
    --llm_alpha 32 \
    --resampler_num_queries 32 \
    --resampler_num_layers 2 \
    --batch_size 4 \
    --logp_batch_size 4 \
    --max_length 160 \
    --freeze_vm True \
    --num_workers 8 \
    --devices 1 \
    --strategy auto \
    --accelerator gpu \
    --precision bf16-mixed \
    2>&1 | tee "${log}"
