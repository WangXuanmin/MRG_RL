#!/bin/bash
set -euo pipefail

cd /data/wang.xuanmin/R2GenGPT

delta_file="save/iu_xray/qwen_dino_sft_8gpu_continue2/checkpoints/checkpoint_epoch1_step190_train.pth"
output_root="save/iu_xray/decode_sweep_sft_continue2"

mkdir -p "${output_root}"

python -u tools/run_iuxray_decode_sweep.py \
  --delta_file "${delta_file}" \
  --output_root "${output_root}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES:-0}" \
  "$@"
