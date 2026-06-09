# MRG_RL: Medical Report Generation Optimization Experiments

This repository contains our IU X-Ray medical report generation experiments for improving report-level language overlap and clinical faithfulness.

The target metrics for the final MRG model were:

- `Bleu_4 >= 0.20`
- `ROUGE_L >= 0.40`
- `CheXbert micro F1 >= 0.40`
- `RadGraph rg_er >= 0.39`

This is not a clean fork or reproduction package for the original R2GenGPT paper. The working code kept some legacy class/file names, such as `models/R2GenGPT.py`, only because the local training and inference scripts were built on that interface. The actual work in this repository is our own experiment stack: Qwen/DINO-based SFT, failed DPO diagnosis, retrieval candidate construction, oracle analysis, clinical metric evaluation, candidate reranking, template baselines, and candidate-copy/number-selection trials.

## What This Repository Contains

Core experiment work:

- IU X-Ray SFT/DPO/RL-style trial records and failure analysis.
- B4/Rouge-focused template baselines and decode sweeps.
- Top-k retrieval candidate construction using image-nearest reports.
- Reference-aware and clinical-aware oracle studies for top-k candidate pools.
- CheXbert and RadGraph evaluation tooling.
- Candidate report reranking attempts:
  - DINO/scalar reranker
  - image-label-guided selector
  - BERT-text cross-reranker
  - model log-prob candidate selector
  - candidate-number selection SFT
- GitHub-ready result artifacts, prediction JSON files, and experiment summaries.

Large checkpoints are intentionally excluded. See `CHECKPOINTS_NOT_INCLUDED.md`.

## Main Finding

The retrieval candidate pool is strong enough to meet the requested targets if the correct candidate can be selected:

| Setting | Test Bleu_4 | Test ROUGE_L | CheXbert micro F1 | RadGraph rg_er |
| --- | ---: | ---: | ---: | ---: |
| Top-50 clinical oracle | 0.3258 | 0.5287 | 0.7849 | 0.4798 |
| Top-20 clinical copy oracle | 0.2556 | 0.4671 | 0.7111 | not rerun |
| Fixed template baseline | 0.1824 | 0.4001 | 0.0000 | 0.3965 |

However, deployable selectors trained in this small-data setting did not reliably select the correct retrieved report. The strongest conclusion is that the next useful step is not more prompt-level tuning, but a better radiology image-text aligned retriever/reranker, preferably trained with larger image-report data such as MIMIC-CXR.

## Important Files

Start here:

- `save/iu_xray/model_registry/EXPERIMENT_LOG.md`
- `save/iu_xray/model_registry/B4_ROUGE_EXPERIMENT.md`
- `save/iu_xray/model_registry/IUXRAY_TRIAL_AND_ERROR_REVIEW.md`
- `README_CORE_PACK.md`
- `PACK_MANIFEST.json`

Core tools:

- `tools/evaluate_iuxray_clinical_metrics.py`
- `tools/build_iuxray_rag_annotation.py`
- `tools/run_iuxray_clinical_oracle.py`
- `tools/build_iuxray_copy_oracle_annotation.py`
- `tools/build_iuxray_candidate_number_annotation.py`
- `tools/run_iuxray_candidate_number_inference.py`
- `tools/train_iuxray_text_cross_reranker.py`
- `tools/train_iuxray_label_guided_selector.py`

Key result folders:

- `save/iu_xray/manual_template_target_b4rouge/`
- `save/iu_xray/rag_top50_clinical_oracle/`
- `save/iu_xray/rag_top20_copy_oracle_cw1/`
- `save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5_test_beam3/`
- `save/iu_xray/qwen_dino_sft_rag_top20_number_cw1_lr5e5_infer_beam1/`
- `save/iu_xray/text_cross_reranker_top50_cw0/`

## Repository Layout

```text
configs/                 Training and inference configuration
dataset/                 IU X-Ray parsing and retrieved-context loading
models/                  MRG model wrapper and vision resampler
tools/                   Experiment builders, evaluators, rerankers, and utilities
scripts/                 Reproducibility shell scripts
data/iu_xray/            Lightweight annotations used by the experiments
save/iu_xray/            Selected summaries, metrics, predictions, and logs
save/iu_xray/model_registry/
                         Human-readable experiment registry and trial notes
```

## Notes On Legacy Names

Some paths still contain `R2GenGPT` or `qwen_dino_sft_*` because those were the active filenames in the remote experiment workspace. They should be read as compatibility names in our MRG experiment codebase, not as a claim that this repository is the upstream R2GenGPT project.

The model stack used in these experiments was changed during the work, including Qwen-based language modeling, DINO visual features, retrieval-context prompts, custom SFT/DPO utilities, candidate reranking tools, and clinical metric evaluation.

## Current Status

The repository is an experiment archive and reproducibility package. It documents both successful upper-bound analyses and failed deployable approaches. The most actionable next direction is:

1. Train or import a radiology image-text aligned retrieval model.
2. Rerank top-k candidate reports with that aligned model.
3. Use candidate copy/fusion only after selection quality is strong enough.
4. Re-evaluate with B4, ROUGE-L, CheXbert micro F1, and RadGraph rg_er.
