#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="data/iu_xray/images"
delta_file="save/iu_xray/qwen_dino_sft/checkpoints/checkpoint_epoch0_step1531_bleu0.105438_cider0.195595.pth"
version="qwen_dino_sft_8gpu_continue2"
savepath="./save/${dataset}/${version}"

mkdir -p "${savepath}"

python -u train.py \
    --stage sft \
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
    --batch_size 4 \
    --val_batch_size 4 \
    --max_epochs 2 \
    --max_length 160 \
    --min_new_tokens 40 \
    --max_new_tokens 160 \
    --learning_rate 5e-5 \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 8 \
    --strategy ddp \
    --accelerator gpu \
    --precision bf16-mixed \
    --limit_val_batches 0 \
    --num_sanity_val_steps 0 \
    2>&1 | tee ${savepath}/log.txt
