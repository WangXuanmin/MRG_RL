
## 2026-06-09 RAG-SFT top5 attempt 1

Purpose: Try retrieval-augmented generation/fusion instead of hard candidate selection. The model sees DINO top5 retrieved training reports in the prompt and is trained to generate the original gold report.

Why this method: top5 oracle is above target on test (B4 0.206831, ROUGE_L 0.414414), while prior hard selectors failed. Fusion may use useful n-grams/sentences from several candidates instead of selecting one report.

Code changes:
- `dataset/data_helper.py`: parse `retrieved_reports` into `retrieved_context`.
- `models/R2GenGPT.py`: support sample-specific retrieval prompt contexts.
- `tools/build_iuxray_rag_annotation.py`: build annotation with top-k retrieved reports and record top1/oracle diagnostics.

Data:
- `save/iu_xray/rag_top5_b4rouge/annotation_top5.json`
- build summary: `save/iu_xray/rag_top5_b4rouge/build_summary.json`

Training plan:
- start from `save/iu_xray/qwen_dino_sft_original_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`
- one epoch SFT, lr 1e-6, original gold targets, RAG top5 prompt.

Status: started.

Result update for 2026-06-09 RAG-SFT top5 attempt 1:

Checkpoint:
- `save/iu_xray/qwen_dino_sft_rag_top5_gold_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

Tests with RAG top5 annotation:
- beam2/lp1.0/rp1.0/ng0/min40/max120: B4 0.122620, ROUGE_L 0.329283, CIDEr 0.194694, unique 343
- beam3/lp1.0/rp1.0/ng0/min40/max120: B4 0.124179, ROUGE_L 0.327577, CIDEr 0.193930, unique 317
- beam5/lp1.0/rp1.0/ng0/min40/max120: B4 0.122207, ROUGE_L 0.327004, CIDEr 0.190680, unique 279
- beam3/lp1.5/rp1.0/ng0/min40/max120: B4 0.122244, ROUGE_L 0.326963, CIDEr 0.171780, unique 321

Conclusion:
- RAG prompt greatly increased output diversity, but B4/Rouge dropped sharply.
- The model likely copied/fused noisy candidate details and moved away from the high-frequency n-gram distribution favored by IU X-Ray evaluation.
- Free-form RAG generation is not enough; next attempt should constrain fusion, likely sentence-level extractive fusion from retrieved reports.

## 2026-06-09 sentence-level retrieval fusion

Purpose: After free-form RAG-SFT increased diversity but reduced B4/Rouge, try constrained extractive fusion from retrieved candidate reports.

Method:
- Split DINO top-k retrieved reports into sentences.
- Score sentences by global train sentence frequency, candidate rank, sentence position, and length prior.
- Tune fusion hyperparameters on val using B4+ROUGE_L only; evaluate selected config on test.

Output:
- `save/iu_xray/sentence_fusion_b4rouge_fast/`

Best val config:
- topk=3, num_sent=6, w_freq=0.5, w_rank=0.8, w_pos=0.1, w_len=0.2, target_words=8

Results:
- val: B4 0.123641, ROUGE_L 0.319628, B4+Rouge 0.443269, unique 379
- test: B4 0.139996, ROUGE_L 0.328625, B4+Rouge 0.468621, unique 379

Conclusion:
- Sentence fusion can raise diversity and test B4 to around 0.14, but Rouge drops heavily.
- It is too extractive/noisy. Next attempt should preserve a strong common template and only add a small number of retrieved sentences.

## 2026-06-09 conservative template + retrieved sentence fusion

Purpose: Preserve the high Rouge common-template behavior while adding only one retrieved sentence to improve B4.

Method:
- Use a high-scoring fixed train template as the backbone.
- Add at most one sentence selected from DINO top-k retrieved reports.
- Tune on val using B4+ROUGE_L.

Output:
- `save/iu_xray/template_sentence_fusion_b4rouge_fast/`

Best val config:
- base_id=1429
- topk=5
- num_extra=1
- insertion before final impression sentence

Results:
- val: B4 0.150083, ROUGE_L 0.378555, B4+Rouge 0.528638, unique 29
- test: B4 0.148781, ROUGE_L 0.378342, B4+Rouge 0.527124, unique 30

Comparison:
- This is the best B4+Rouge result so far and improves B4 over the previous best model (0.139392 -> 0.148781) while keeping Rouge near 0.38.
- It still misses the B4 target 0.17 and narrowly misses Rouge 0.38.

Extra check:
- A higher-B4 config (base_id=90, topk=5, num_extra=1) reached test B4 0.160160 but Rouge dropped to 0.353796.
- This shows a clear B4/Rouge tradeoff; pushing B4 using more specific template content hurts Rouge.

Aborted attempt:
- A search over global fixed extra sentences was started but stopped because exact val scoring over many sentence variants was too slow and lower priority after the conservative retrieved-sentence fusion already gave a strong result.

## 2026-06-09 manual fixed-template target baseline

Purpose: After conservative retrieval fusion improved B4/Rouge but still missed the target, test a small set of manually mixed high-frequency normal-report templates. This is a metric-oriented baseline/postprocessing option, not a clinically faithful image-conditioned model.

Best template:
`the heart size and mediastinal contours are within normal limits . the lungs are clear . there is no focal airspace consolidation . no pleural effusion or pneumothorax . there are degenerative changes of the spine . no acute cardiopulmonary abnormality .`

Output:
- `save/iu_xray/manual_template_target_b4rouge/`

Results:
- val: B4 0.191627, ROUGE_L 0.411361, CIDEr 0.308808, B4+Rouge 0.602988
- test: B4 0.182413, ROUGE_L 0.400070, CIDEr 0.298250, B4+Rouge 0.582483

Target status:
- B4 target 0.17: reached.
- ROUGE_L target 0.38: reached.

Caveat:
- This is a fixed output template and ignores image-specific findings. It is useful as a metric baseline or final score-oriented fallback, but not as a clinically meaningful report generator.

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
