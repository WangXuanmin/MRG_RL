# R2GenGPT Qwen-DINO-DPO Upgrade

## Implemented mainline changes

- Vision encoder: replaced Swin with Hugging Face `AutoModel`, defaulting to DINOv2.
- Visual compression: added `models/vision_resampler.py` with a Perceiver/Q-Former style learnable-query cross-attention resampler.
- Language model: replaced LLaMA-specific code with `AutoModelForCausalLM` + `AutoTokenizer`, defaulting to Qwen3-8B.
- Training: Qwen LoRA is enabled by default.
- Stages:
  - `--stage sft`: supervised report generation.
  - `--stage dpo`: preference optimization from `id/chosen/rejected` JSONL.
- Preference construction:
  - `build_preferences.py` combines n-gram metrics, optional RadGraph/CheXbert scores, and optional DeepSeek judge scores.
- Clinical scoring:
  - `compute_clinical_scores.py` computes RadGraph F1 directly if the `radgraph` package is installed.
  - It exports CheXbert CSV inputs and can compute CheXbert micro-F1 from CheXbert labeler outputs.

## Resource paths

- DINOv2: `resources/models/facebook_dinov2-base`
- RadGraph-XL weights: `resources/models/radgraph/modern-radgraph-xl.tar.gz`
- Qwen3-8B: `resources/models/Qwen3-8B`
- CheXbert checkpoint target: `resources/models/chexbert/chexbert.pth`
- IU-Xray raw files: `resources/datasets/iu_xray`

## Current resource status

- CheXbert checkpoint is downloaded from `StanfordAIMI/RRG_scorers/chexbert.pth`.
- Qwen3-8B is downloaded under `resources/models/Qwen3-8B`.
- IU-Xray raw PNG and reports archives are downloaded under `resources/datasets/iu_xray`.
- IU-Xray has been converted to R2GenGPT annotation format under `data/iu_xray`.

## Typical workflow

Prepare resources:

```bash
./scripts/download_resources.sh
python3 scripts/prepare_iuxray.py \
  --raw_dir resources/datasets/iu_xray \
  --output_dir data/iu_xray
```

SFT cold start:

```bash
bash scripts/qwen_dino_sft_iuxray.sh
```

Generate multiple candidate result files by running test with different decoding settings/checkpoints.

Build preference pairs:

```bash
export DEEPSEEK_API_KEY=your_key

python3 build_preferences.py \
  --refs save/iu_xray/qwen_dino_sft/result/test_refs.json \
  --candidates save/iu_xray/run_a/result/test_result.json save/iu_xray/run_b/result/test_result.json \
  --output data/iu_xray/preferences.jsonl \
  --use_deepseek \
  --ngram_weight 0.3 \
  --deepseek_weight 0.5 \
  --radgraph_weight 0.1 \
  --chexbert_weight 0.1
```

DPO:

```bash
bash scripts/qwen_dino_dpo_iuxray.sh
```

## Important note on DPO

This implementation uses reward scores to rank candidate reports and produce `chosen/rejected` pairs. DPO itself trains from preference pairs, not directly from scalar rewards.
