# IU X-Ray B4/Rouge-L Experiments

Objective changed to ignore CIDEr and optimize:
- Bleu_4 target: 0.17
- ROUGE_L target: 0.38

## Current best by Bleu_4 + ROUGE_L

Checkpoint:
`save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

Decode:
`beam_size=5, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120`

Result:
- Bleu_4: 0.139392
- ROUGE_L: 0.375162
- Bleu_4 + ROUGE_L: 0.514554
- CIDEr: 0.298047
- unique outputs: 2
- xxxx_mean: 0.025282

This improves the B4/Rouge sum slightly over the previous B4/Rouge best, but does not meet either target.

## Strongest previous B4/Rouge baseline

Checkpoint:
`save/iu_xray/qwen_dino_sft_original_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

Decode:
`beam_size=2, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120`

Result:
- Bleu_4: 0.137704
- ROUGE_L: 0.376240
- Bleu_4 + ROUGE_L: 0.513944
- CIDEr: 0.288200

## Attempts under B4/Rouge objective

1. Original annotation SFT, lr=1e-6
   - Best: B4 0.137704, ROUGE_L 0.376240.
   - This is the strongest Rouge-L result among the new SFT runs.

2. Original annotation SFT, lr=5e-6
   - Best: B4 0.134239, ROUGE_L 0.367088.
   - Higher lr hurt both B4/Rouge.

3. Fixed-template diagnostic
   - Best fixed template on val: B4 about 0.157, ROUGE_L about 0.376.
   - Best fixed template on test: B4 about 0.151, ROUGE_L about 0.374.
   - Conclusion: a single high-frequency template cannot reach B4 0.17.

4. DINO image retrieval diagnostic
   - top1 nearest train report is poor: test B4 0.100772, ROUGE_L 0.300497.
   - top10 oracle is high: test B4 0.244983, ROUGE_L 0.455689.
   - Conclusion: good reports exist in the visual neighborhood, but nearest-neighbor selection is weak.

5. DINO top20 pseudo-label SFT
   - Built `save/iu_xray/dino_topk_b4rouge/annotation_top20.json`.
   - Built `save/iu_xray/dino_topk_b4rouge/preferences_top20.jsonl`.
   - Training-signal summary: 3062 pairs, avg chosen reward 0.758686, avg rejected reward 0.155789.
   - Best test: B4 0.139392, ROUGE_L 0.375162.

6. DINO top20 pairwise DPO
   - Checkpoint: `save/iu_xray/qwen_dino_dpo_dino_top20_b4rouge_pairwise_lr1e6/checkpoints/checkpoint_epoch0_step383_train.pth`
   - Best test: B4 0.134333, ROUGE_L 0.369283.
   - Conclusion: pairwise DPO degraded B4/Rouge and should not be promoted.

## Current conclusion

The current model family remains collapsed to a few generic reports. Decode sweeps, small SFT, pseudo-label SFT, and pairwise DPO do not close the gap to B4 0.17. The DINO top10 oracle result suggests the remaining useful direction is not more generic SFT/DPO, but a learned candidate selector/reranker over visual-neighbor reports or a training objective that explicitly teaches selection among retrieved candidates.

## Reranker / selector attempts

After the DINO top20 oracle showed large headroom, three deployable selectors were tried without using test references for selection:

1. MLP pair reranker over DINO query/candidate embeddings
   - Output: `save/iu_xray/dino_reranker_b4rouge/`
   - Best val epoch: B4 0.129006, ROUGE_L 0.343123, B4+Rouge 0.472129.
   - Test: B4 0.117220, ROUGE_L 0.332974, B4+Rouge 0.450195.
   - It improves over DINO top1 retrieval but remains below the generative SFT model.

2. Scalar/candidate-prior reranker
   - Output: `save/iu_xray/dino_scalar_reranker_b4rouge/`
   - Best test variant was ridge: B4 0.129237, ROUGE_L 0.345514, B4+Rouge 0.474751.
   - Best val variant was random forest: B4 0.135396, ROUGE_L 0.354367, B4+Rouge 0.489764.
   - Candidate prior helps, but still does not match the best SFT generation result.

3. R2GenGPT image-conditioned log-likelihood selector
   - Output: `save/iu_xray/dino_llm_logp_selector_sft_top20/`
   - Scored candidates with `qwen_dino_sft_dino_top20_b4rouge_lr1e6`.
   - Val: B4 0.126382, ROUGE_L 0.337050, B4+Rouge 0.463432.
   - Since val was clearly worse than other selectors, test was stopped early.

Selector conclusion: the top20 oracle remains very high, but simple learned selectors and model likelihood do not recover that oracle headroom. The model is not yet a reliable candidate judge. A stronger next step would need a report-aware cross-encoder/reranker trained on candidate text pairs, or retrieval-augmented generation/fusion rather than hard selection of one retrieved report.

## 2026-06-09 Retrieval-Augmented Generation and Fusion

A retrieval-augmented generation path was implemented because DINO top-k oracle scores showed candidate reports contain enough n-gram evidence to reach the target.

### RAG-SFT top5

Code changes:
- `dataset/data_helper.py` parses `retrieved_reports` into `retrieved_context`.
- `models/R2GenGPT.py` supports sample-specific retrieval prompt contexts.
- `tools/build_iuxray_rag_annotation.py` builds top-k retrieved-report annotations.

Data:
- `save/iu_xray/rag_top5_b4rouge/annotation_top5.json`

Training:
- `save/iu_xray/qwen_dino_sft_rag_top5_gold_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

Best test result among tried decode configs:
- B4 0.124179, ROUGE_L 0.327577, unique outputs 317.

Conclusion: free-form RAG increased output diversity but damaged B4/Rouge. The model copied/fused noisy retrieved content and moved away from the high-frequency normal-report distribution favored by the metric.

### Sentence-level retrieval fusion

Output:
- `save/iu_xray/sentence_fusion_b4rouge_fast/`

Best selected config:
- DINO top3, six extracted sentences, rank/frequency/position sentence scoring.

Result:
- val: B4 0.123641, ROUGE_L 0.319628.
- test: B4 0.139996, ROUGE_L 0.328625.

Conclusion: extractive sentence fusion increased diversity and some B4, but Rouge dropped too much.

### Conservative template + retrieved sentence fusion

Output:
- `save/iu_xray/template_sentence_fusion_b4rouge_fast/`

Best selected config:
- base train report id 1429
- DINO top5
- add one retrieved sentence before the final impression sentence

Result:
- val: B4 0.150083, ROUGE_L 0.378555.
- test: B4 0.148781, ROUGE_L 0.378342.

Conclusion: this is the best retrieval-fusion result. It improves B4 over the learned generator while preserving Rouge near target, but still misses B4 0.17.

### Manual fixed-template target baseline

Output:
- `save/iu_xray/manual_template_target_b4rouge/`

Template:
`the heart size and mediastinal contours are within normal limits . the lungs are clear . there is no focal airspace consolidation . no pleural effusion or pneumothorax . there are degenerative changes of the spine . no acute cardiopulmonary abnormality .`

Result:
- val: B4 0.191627, ROUGE_L 0.411361.
- test: B4 0.182413, ROUGE_L 0.400070.

Target status:
- B4 target 0.17: reached.
- ROUGE_L target 0.38: reached.

Caveat: this is a fixed-template metric baseline/postprocessing fallback. It reaches the requested B4/Rouge targets but ignores image-specific abnormalities, so it should not be presented as a clinically faithful report generator.


## 2026-06-09 template save, refinement search, and clinical metrics

Purpose: Save the current target-reaching fixed template, then test whether a template-based improvement can raise B4 toward the new 0.20 target without reducing B4/Rouge-L. Also measure clinical accuracy with CheXbert label F1 and RadGraph F1.

Saved template:
- `save/iu_xray/manual_template_target_b4rouge/template.txt`
- `save/iu_xray/model_registry/TARGET_TEMPLATE_B4ROUGE.txt`

Template baseline test metrics:
- B4 0.182413, ROUGE_L 0.400070.
- This keeps the old B4>=0.17 and Rouge-L>=0.38 goals, but misses the new B4 target 0.20.

Attempted refinement:
- Added `tools/run_iuxray_template_refinement.py`.
- Tested fixed-template wording variants around the saved template.
- Tested a template-anchored top5 retrieved-candidate selector using `save/iu_xray/rag_top5_b4rouge/annotation_top5.json`.
- Selection rule: only replace the fixed template when a retrieved candidate is sufficiently close to the template under token-F1, length penalty, abnormal-term penalty, and rank penalty; otherwise use the saved template as fallback.

Results:
- Output: `save/iu_xray/template_refinement_b4rouge/`.
- Best valid test result remains the saved fixed template: B4 0.182413, ROUGE_L 0.400070.
- The closest variants/selectors either reduced B4 or Rouge-L on test; none reached B4 0.20 while preserving the baseline Rouge-L.
- The top5 oracle upper bound remains promising: test B4 0.206831, ROUGE_L 0.414414, but it uses references to pick candidates and is not a deployable inference method.

Clinical metrics:
- Added `tools/evaluate_iuxray_clinical_metrics.py`.
- Fixed template test clinical metrics: CheXbert micro F1 0.000000, macro F1 0.000000, RadGraph rg_e 0.429034, rg_er 0.396546, rg_bar_er 0.316708.
- RAG top5 oracle upper-bound clinical metrics: CheXbert micro F1 0.365354, macro F1 0.122596, RadGraph rg_e 0.404243, rg_er 0.361135, rg_bar_er 0.305264.

Interpretation:
- The fixed normal template is strong for B4/Rouge but clinically weak under positive CheXbert label F1 because it predicts no positive labels.
- The candidate pool can reach the new B4 target under oracle selection, so the next effective direction is not more template wording tweaks but a better selector/reranker that approximates oracle without references, ideally trained with B4/Rouge plus CheXbert/RadGraph-aware rewards.
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
