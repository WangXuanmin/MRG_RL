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

The central finding is that the target is reachable in the retrieved-candidate space, but not yet reachable with the deployable selectors and generators trained in this small-data setting. In other words, the bottleneck moved from "can we find a good report?" to "can the model reliably select or use that report without reference access?"

| Setting | Test Bleu_4 | Test ROUGE_L | CheXbert micro F1 | RadGraph rg_er |
| --- | ---: | ---: | ---: | ---: |
| Top-50 clinical oracle | 0.3258 | 0.5287 | 0.7849 | 0.4798 |
| Top-20 clinical copy oracle | 0.2556 | 0.4671 | 0.7111 | not rerun |
| Fixed template baseline | 0.1824 | 0.4001 | 0.0000 | 0.3965 |

What we tried and found:

| Experiment line | Best/representative result | Finding |
| --- | --- | --- |
| Original SFT/decode sweeps | Best normal SFT tests stayed around `B4≈0.13-0.14`, `Rouge-L≈0.37` | Standard generation did not reach the language-overlap target. Decode tuning alone was not enough. |
| DPO variants | DPO did not improve the target metrics and was diagnosed as unstable/misaligned for this setup | Pairwise preference training was not useful without a stronger reward/selection signal. |
| Fixed template baseline | `B4=0.1824`, `Rouge-L=0.4001`, `CheXbert F1=0.0000`, `RadGraph rg_er=0.3965` | A generic normal-report template is a strong B4/Rouge baseline on IU X-Ray, but it fails clinical label accuracy. |
| Template refinement and sentence fusion | Could preserve Rouge-L near `0.40`, but did not raise B4 to `0.20` or improve CheXbert meaningfully | Template methods exploit dataset bias, but cannot solve abnormal clinical correctness. |
| Top-k retrieval oracle | Top-50 oracle: `B4=0.3645`, `Rouge-L=0.5624`; clinical oracle also passed CheXbert/RadGraph | Retrieval candidates contain enough information to meet all targets if selected correctly. |
| Clinical scalar / image-label selector | Test candidate selector reached `CheXbert F1=0.4017`, but only `B4=0.1284`, `Rouge-L=0.3473` | Clinical signal can improve labels, but current image features select reports with poor ngram overlap. |
| Text cross-reranker | Best test: `B4=0.1350`, `Rouge-L=0.3528`, `CheXbert F1=0.3551` | Frozen text embeddings plus DINO image embeddings were not enough for reliable candidate selection. |
| Template-candidate hybrid | Any nonzero replacement of the template with selected candidates reduced B4/Rouge | Weak selectors damage the strong template baseline rather than improving it. |
| Copy-oracle SFT with top-20 candidates in prompt | Free generation: `B4=0.1111`, `Rouge-L=0.3152`, `CheXbert F1=0.1905`, `RadGraph rg_er=0.2689` | Even when trained to copy oracle candidates, the model did not reliably use long retrieved context during generation. |
| Model log-prob candidate selection | `B4=0.0632`, `Rouge-L=0.2735`; average selected rank `9.69` | Sequence likelihood was dominated by language prior/style, not image-conditioned correctness. |
| Candidate-number SFT | Best tested checkpoint: `B4=0.1068`, `Rouge-L=0.3256`; output biased toward high candidate numbers | Reducing generation to 20-way candidate-number selection still did not learn robust image-conditioned ranking. |

Overall, the useful next step is not more prompt-level tuning on IU X-Ray alone. The next model improvement should replace the weak selection signal with a radiology image-text aligned retriever/reranker, preferably trained or adapted with larger image-report data such as MIMIC-CXR, and then use candidate copy/fusion only after selection quality is strong.

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
