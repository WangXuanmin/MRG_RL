#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="data/iu_xray/images"
delta_file="save/iu_xray/qwen_dino_sft_8gpu_continue2/checkpoints/checkpoint_epoch1_step190_train.pth"
outdir="save/iu_xray/candidates"

mkdir -p "${outdir}"

common_args=(
  --split train
  --stage sft
  --dataset ${dataset}
  --annotation ${annotation}
  --base_dir ${base_dir}
  --delta_file ${delta_file}
  --vision_model resources/models/facebook_dinov2-base
  --llm_model resources/models/Qwen3-8B
  --llm_use_lora True
  --llm_r 16
  --llm_alpha 32
  --resampler_num_queries 32
  --resampler_num_layers 2
  --test_batch_size 8
  --max_length 160
  --min_new_tokens 40
  --max_new_tokens 160
  --freeze_vm True
  --num_workers 8
  --devices 1
  --strategy auto
  --accelerator gpu
  --precision bf16-mixed
)

CUDA_VISIBLE_DEVICES=0 python -u generate_candidates.py "${common_args[@]}" \
  --beam_size 3 --do_sample False --temperature 0 \
  --output "${outdir}/train_beam.json" \
  --refs_output "${outdir}/train_refs.json" \
  2>&1 | tee "${outdir}/train_beam.log" &
pid_beam=$!

CUDA_VISIBLE_DEVICES=1 python -u generate_candidates.py "${common_args[@]}" \
  --beam_size 1 --do_sample True --temperature 0.7 \
  --output "${outdir}/train_sample_t07.json" \
  2>&1 | tee "${outdir}/train_sample_t07.log" &
pid_t07=$!

CUDA_VISIBLE_DEVICES=2 python -u generate_candidates.py "${common_args[@]}" \
  --beam_size 1 --do_sample True --temperature 1.0 \
  --output "${outdir}/train_sample_t10.json" \
  2>&1 | tee "${outdir}/train_sample_t10.log" &
pid_t10=$!

wait ${pid_beam}
wait ${pid_t07}
wait ${pid_t10}
