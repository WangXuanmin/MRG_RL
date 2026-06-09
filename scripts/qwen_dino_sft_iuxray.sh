#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="data/iu_xray/images"
version="qwen_dino_sft"
savepath="./save/${dataset}/${version}"

mkdir -p "${savepath}"

python -u train.py \
    --stage sft \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --vision_model resources/models/facebook_dinov2-base \
    --llm_model resources/models/Qwen3-8B \
    --llm_use_lora True \
    --llm_r 16 \
    --llm_alpha 32 \
    --resampler_num_queries 32 \
    --resampler_num_layers 2 \
    --batch_size 2 \
    --val_batch_size 4 \
    --max_epochs 2 \
    --max_length 160 \
    --min_new_tokens 40 \
    --max_new_tokens 160 \
    --learning_rate 1e-4 \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 1 \
    --strategy auto \
    2>&1 | tee ${savepath}/log.txt
