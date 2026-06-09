#!/usr/bin/env python3
from pathlib import Path


TEXT = r"""

## 2026-06-09 top-k retrieval/copy/number-selection trials for new target

Target requested for final MRG model:
`Bleu_4 >= 0.20`, `ROUGE_L >= 0.40`, `CheXbert micro F1 >= 0.40`, `RadGraph rg_er >= 0.39`.

### 1. Top-50 and top-20 candidate-space upper bounds

Method:
- Built top-50 retrieved-report annotation from DINO nearest train studies.
- Computed reference-aware oracle over retrieved candidates.
- Added clinical-aware oracle using score `B4 + Rouge-L + chex_weight * CheXbert_F1`.

Key results:
- Top-50 ngram oracle on test: `B4=0.364463`, `Rouge-L=0.562369`.
- Top-50 clinical oracle `cw=1.0` on test: `B4=0.325783`, `Rouge-L=0.528731`, `CheXbert micro F1=0.784916`, `RadGraph rg_er=0.479829`.
- Top-20 clinical copy oracle `cw=1.0` on test: `B4=0.255570`, `Rouge-L=0.467124`, `CheXbert micro F1=0.711111`.

Conclusion:
- The retrieval candidate pool contains reports that can meet all four targets.
- The bottleneck is not candidate availability; it is selecting/generating the right candidate without reference access.

### 2. Clinical scalar/image-label selector

Method:
- Trained image-side CheXbert label predictors from DINO embeddings.
- Used predicted labels to rerank top-50 candidates by expected label F1, blended with DINO similarity/rank/ngram scalar model.
- Also tested label-driven template generation.

Result:
- Image label classifier generalized poorly: train CheXbert micro F1 `0.7927`, val `0.3036`, test `0.3012`.
- Best candidate selector on test: `B4=0.128421`, `Rouge-L=0.347326`, `CheXbert micro F1=0.401734`.
- Best label template on test: `B4=0.124345`, `Rouge-L=0.345060`, intended CheXbert F1 `0.281787`.

Problem:
- DINO embeddings alone are not sufficient for robust clinical label prediction on this small IU X-Ray split.
- Adding clinical signal can lift CheXbert slightly, but damages B4/Rouge because the selected candidate text is not reference-like.

### 3. Text semantic cross-reranker

Method:
- Encoded candidate reports with local BERT embeddings.
- Trained a listwise MLP scorer over image embedding, candidate text embedding, and scalar retrieval features.
- Supervision was top-50 candidate reward `B4+Rouge` with optional clinical reward.

Result (`chex_weight=0`, best val epoch 16):
- Val: `B4=0.143998`, `Rouge-L=0.367749`, `CheXbert F1=0.322289`.
- Test: `B4=0.134997`, `Rouge-L=0.352760`, `CheXbert F1=0.355140`.

Problem:
- Text embeddings helped only slightly over scalar baselines.
- The model still did not learn the image-to-report semantic match needed to select from top-50.

### 4. Template-candidate hybrid fallback

Method:
- Started from the strong fixed template (`test B4=0.182413`, `Rouge-L=0.400070`).
- Replaced a small fraction of samples with current best selector candidates, using report length as a crude confidence proxy.

Result:
- Any nonzero replacement fraction reduced B4/Rouge.
- Example best nonzero setting around 3% replacement: `B4≈0.1806`, `Rouge-L≈0.3966`.

Problem:
- Wrong candidate substitutions quickly break the high-Rouge generic template.
- Hybrid fallback is only viable with a much stronger confidence/selection signal.

### 5. Copy-oracle SFT with retrieved reports in prompt

Method:
- Built `data/iu_xray/annotation_rag_top20_copy_oracle_cw1.json`.
- Train split target report was replaced by the top-20 clinical oracle candidate (`B4+Rouge+CheXbert`).
- Val/test retained original references; prompt included top-20 retrieved reports.
- Continued from `qwen_dino_sft_8gpu_continue2/checkpoint_epoch1_step190_train.pth`.

Upper bound of the training/inference candidate space:
- Test top-20 copy oracle: `B4=0.255570`, `Rouge-L=0.467124`, `CheXbert F1=0.711111`.

Free-generation result:
- Checkpoint: `save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5/checkpoints/checkpoint_epoch1_step766_train.pth`
- Test beam3: `B4=0.111095`, `Rouge-L=0.315158`, `CheXbert micro F1=0.190476`, `RadGraph rg_er=0.268899`.

Problem:
- The model did not learn to copy/fuse the oracle candidate despite seeing top-20 reports.
- Long retrieved context plus long report generation appears too hard/unstable for this small dataset.

### 6. R2GenGPT conditional log-prob candidate selection

Method:
- Used the copy-oracle SFT checkpoint.
- At inference, scored each retrieved candidate report by conditional average token log-prob under R2GenGPT with the same top-20 context.
- Selected the highest log-prob candidate instead of free-generating.

Result:
- Test: `B4=0.063173`, `Rouge-L=0.273543`, `CIDEr=0.078336`.
- Average selected rank: `9.69`; top-1 selected only `4.97%`.

Problem:
- The model's sequence likelihood is dominated by language prior and length/style preference, not candidate correctness.
- This approach is worse than free generation and much worse than the generic template.

### 7. Candidate-number SFT

Method:
- Added configurable `--retrieval_instruction` to R2GenGPT so retrieval prompts can ask for candidate selection instead of report generation.
- Built `data/iu_xray/annotation_rag_top20_number_cw1.json`.
- Train target became only the oracle candidate number (`1 .` to `20 .`).
- Inference generated a number, parsed it, and copied that retrieved report.

Results:
- Epoch 0: `B4=0.104615`, `Rouge-L=0.325874`, avg rank `10.96`, top1 rate `1.57%`.
- Epoch 1: `B4=0.106751`, `Rouge-L=0.325570`, avg rank `11.14`, top1 rate `12.57%`.
- Epoch 2: `B4=0.106038`, `Rouge-L=0.321158`, avg rank `12.55`, top1 rate `6.28%`.

Problem:
- The task was reduced to 20-way selection, but the model still learned a biased number prior rather than image-conditioned candidate ranking.
- Raw output distribution was concentrated on high candidate numbers such as 10-19.

### Current diagnosis

The candidate oracle proves that target scores are attainable if the right retrieved report is selected. However, every deployable selector tried so far fails:
- DINO-only selectors lack clinical/semantic alignment.
- Frozen BERT text embeddings plus DINO image embeddings are not enough.
- R2GenGPT does not reliably use long top-k retrieved context for selection, even when the target is only a number.

Most likely next improvement:
- Replace DINO-only retrieval/reranking with a radiology image-text aligned retriever/reranker.
- Train a contrastive image-report model or cross-encoder on IU X-Ray plus larger MIMIC-CXR image-report pairs, then use it to rerank top-k candidates before generation/copy.
- For small IU X-Ray alone, a fixed template still remains the strongest non-oracle B4/Rouge baseline, but it cannot meet the new CheXbert target.
"""


def append(path):
    p = Path(path)
    old = p.read_text() if p.exists() else ""
    if "Candidate-number SFT" in old and "2026-06-09 top-k retrieval/copy/number-selection trials" in old:
        return
    p.write_text(old.rstrip() + "\n" + TEXT.strip() + "\n")


def main():
    append("save/iu_xray/model_registry/EXPERIMENT_LOG.md")
    append("save/iu_xray/model_registry/B4_ROUGE_EXPERIMENT.md")
    append("save/iu_xray/model_registry/IUXRAY_TRIAL_AND_ERROR_REVIEW.md")
    print("appended current experiment log")


if __name__ == "__main__":
    main()
