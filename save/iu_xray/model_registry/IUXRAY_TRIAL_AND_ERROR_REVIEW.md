# IU X-Ray MRG_RL 试错复盘：SFT / DPO / 检索重排

日期：2026-06-09  
历史远端工作路径：`/data/wang.xuanmin/R2GenGPT`  
当前目标：不再关注 CIDEr，主要提升 `Bleu_4` 和 `ROUGE_L`。

目标值：

- `Bleu_4 >= 0.17`
- `ROUGE_L >= 0.38`

当前最好结果仍未达标：

- checkpoint: `save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`
- decode: `beam_size=5, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120`
- test: `Bleu_4=0.139392, ROUGE_L=0.375162, CIDEr=0.298047`

## 1. 原始 DPO 失败与清理

最早尝试过一版 DPO，训练和测试结果都不理想。该版本的主要问题是 DPO 目标过于粗糙，容易把模型推向偏离 SFT 分布的方向，而且没有足够稳定的 reference anchor。

已清理内容：

- 删除旧 DPO 脚本：
  - `scripts/qwen_dino_dpo_iuxray.sh`
  - `scripts/qwen_dino_dpo_iuxray_8gpu.sh`
- 删除旧失败 DPO 权重：
  - `save/iu_xray/qwen_dino_dpo_8gpu/checkpoints/checkpoint_epoch0_step366_train.pth`

保留内容：

- 原 DPO 测试结果文件仍保留，用于对比和追溯：
  - `save/iu_xray/qwen_dino_dpo_8gpu_test/result/test_result.json`
  - `save/iu_xray/qwen_dino_dpo_8gpu_test/result/test_refs.json`
  - 对应 test log

这一步的经验是：失败权重可以清掉，但结果文件要保留，因为后面判断“新 DPO 是否真的改进”需要历史对照。

## 2. DPO 实现层面的改进

为了避免重走原始 DPO 的失败路径，先对 DPO 代码做了几个稳定性改造。

改动包括：

- `configs/config.py`
  - 新增 `--dpo_objective`
  - 新增 `--dpo_sft_loss_weight`
  - 新增 `--dpo_require_ref_logps`
- `dataset/data_helper.py`
  - 支持 preference row 中的 `ref_chosen_logp` / `ref_rejected_logp`
- `models/R2GenGPT.py`
  - 支持 reference-anchored DPO
  - 支持 DPO loss 中加入 SFT loss anchor
- 新增 reference logp 预计算脚本和 safe DPO 脚本

这些改动解决的是“DPO 训练形式不稳”的问题，但没有直接解决模型输出高度模板化的问题。

后续不足：

- reference DPO 需要预计算 ref logps，流程更重。
- pairwise DPO 即使加 SFT anchor，仍可能把输出推坏。
- DPO 的效果高度依赖 preference pair 的质量。如果 chosen/rejected 只是按 n-gram 指标构造，模型可能学到“更像模板”，而不是学到图像相关差异。

## 3. 生成参数 bug 修复

早期发现 `no_repeat_ngram_size` 配置没有真正传进 `self.llm.generate(...)`。

修复：

- 在 `models/R2GenGPT.py` 的 generation 调用中加入：
  - `no_repeat_ngram_size=self.hparams.no_repeat_ngram_size`

修复后重新做 decode sweep。结果显示：

- `no_repeat_ngram_size=2` 往往会压低 B4/Rouge。
- IU X-Ray 报告本身包含大量高频医学短语，强行禁止 n-gram 重复会破坏正常报告模板。

结论：

- 这个 bug 必须修，但修完后并没有带来 B4/Rouge 的大幅提升。
- 对当前数据集，`no_repeat_ngram_size=0` 通常比 `2` 更适合 B4/Rouge。

## 4. 重新定义目标：从 CIDEr 转向 B4/Rouge

用户明确指出 CIDEr 可以不管，主要目标是：

- B4 到 `0.17`
- Rouge-L 到 `0.38`

因此重新扫描历史结果，不再按 CIDEr 或原 combined score 排名。

历史上比较强的 SFT baseline：

- checkpoint: `save/iu_xray/qwen_dino_sft_8gpu_continue2/checkpoints/checkpoint_epoch1_step190_train.pth`
- old test: `Bleu_4=0.138284, ROUGE_L=0.340637, CIDEr=0.282538`

修复 no-repeat 后的 decode sweep 中，B4 最高的配置：

- output: `save/iu_xray/decode_sweep_sft_continue2/beam3_lp1p5_rp1p0_ng0_min40_max120`
- `Bleu_4=0.139409`
- `ROUGE_L=0.341329`
- `CIDEr=0.292176`

问题：

- B4 最高能到约 `0.139`，但 Rouge-L 仍低。
- 单靠 decode 参数很难把 B4 拉到 `0.17`。

## 5. 原始 annotation 继续 SFT

为了提升 Rouge-L，先从已有 SFT checkpoint 出发，在原始 annotation 上继续小学习率 SFT。

### 5.1 lr=1e-6

checkpoint:

- `save/iu_xray/qwen_dino_sft_original_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

最佳配置：

- `beam_size=2, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120`

结果：

- `Bleu_4=0.137704`
- `ROUGE_L=0.376240`
- `CIDEr=0.288200`

优点：

- Rouge-L 接近 `0.38`。

问题：

- B4 仍只有 `0.138` 左右，离 `0.17` 很远。
- 输出仍然明显模板化。

### 5.2 lr=5e-6

checkpoint:

- `save/iu_xray/qwen_dino_sft_original_b4rouge_lr5e6/checkpoints/checkpoint_epoch0_step95_train.pth`

最佳结果：

- `Bleu_4=0.134239`
- `ROUGE_L=0.367088`
- `CIDEr=0.317139`

结论：

- 加大学习率没有提升 B4/Rouge，反而损伤效果。
- 继续在原始 annotation 上做更激进 SFT 不是好方向。

## 6. 固定模板上界诊断

为了判断“模型塌缩到通用模板”是否有可能达标，做了固定模板诊断。

方法：

1. 从 train/val 报告中提取去重后的报告文本。
2. 假设模型对所有测试样本都输出同一条固定报告。
3. 计算这条固定报告在 val/test 上的 corpus B4/Rouge。

结果：

- val 固定模板最高：
  - B4 约 `0.157`
  - Rouge-L 约 `0.376`
- test 固定模板最高：
  - B4 约 `0.151`
  - Rouge-L 约 `0.374`

结论：

- 单一高频模板无法达到 `B4=0.17`。
- 如果模型继续只输出 2-3 个通用正常报告模板，B4 的天花板很低。

后续改进方向：

- 必须让输出和图像/病例差异相关。
- 不能只优化“更像常见正常报告”。

## 7. DINO 图像检索诊断

为了判断图像特征里是否存在可用信息，用冻结 DINOv2 特征做图像近邻检索。

方法：

1. 对 train/val/test 的图像用 DINOv2 抽 embedding。
2. 对 val/test 样本，在 train 中找 top-k 视觉近邻。
3. 用近邻报告作为候选输出。

### 7.1 top1 最近邻

test top1 最近邻结果：

- `Bleu_4=0.100772`
- `ROUGE_L=0.300497`
- `CIDEr=0.203462`

结论：

- 直接选视觉最近邻很差。
- DINO 图像相似不等价于报告文本相似。

### 7.2 top10/top20 oracle

oracle 的定义：

- 对每个测试样本，先取 DINO top-k 近邻报告作为候选。
- 然后“偷偷看参考答案”，在 top-k 里选 `B4 + ROUGE_L` 最高的候选。
- 这不是可部署方法，只是诊断候选池上界。

结果：

- test top10 oracle:
  - `Bleu_4=0.244983`
  - `ROUGE_L=0.455689`
  - `CIDEr=0.890580`
- test top20 oracle:
  - `Bleu_4=0.288231`
  - `ROUGE_L=0.493809`
  - `CIDEr=1.171637`

结论：

- top-k 候选池中确实经常包含很好的报告。
- 主要问题变成：如何不用参考答案，学会从 top-k 候选中选出好报告。

## 8. DINO top20 伪标签 SFT

根据 oracle 诊断，尝试把 top20 候选中的高分报告作为训练目标。

方法：

1. 对每个训练样本找 DINO top20 近邻。
2. 用该训练样本自己的 gold report 计算候选的 `B4 + ROUGE_L`。
3. 选分数最高的候选报告作为 pseudo target。
4. 生成新的 annotation 和 preference 文件。

生成文件：

- `save/iu_xray/dino_topk_b4rouge/annotation_top20.json`
- `save/iu_xray/dino_topk_b4rouge/preferences_top20.jsonl`
- `save/iu_xray/dino_topk_b4rouge/build_summary.json`

构造数据的统计：

- pairs: `3062`
- avg chosen reward: `0.758686`
- avg rejected reward: `0.155789`
- avg margin: `0.602896`

训练：

- checkpoint: `save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

最佳测试结果：

- decode: `beam_size=5, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120`
- `Bleu_4=0.139392`
- `ROUGE_L=0.375162`
- `CIDEr=0.298047`

改进：

- `B4 + Rouge` 总分略高于原始 SFT。
- 当前按 B4/Rouge 排序的最好模型。

不足：

- 提升非常小。
- 没有把 top20 oracle 的潜力学出来。
- 输出仍然只有很少数模板，unique outputs 很低。
- 说明“把 oracle 候选当 pseudo label 做 SFT”并不能自然教会模型做视觉相关选择。

## 9. DINO top20 pairwise DPO

基于同一批 top20 preference，尝试 DPO。

方法：

- chosen: DINO top20 中 `B4 + ROUGE_L` 最高的报告
- rejected: DINO top20 中最低的报告
- 起点: DINO top20 pseudo-label SFT checkpoint
- DPO objective: pairwise
- SFT anchor: `dpo_sft_loss_weight=0.05`
- beta: `0.05`

checkpoint:

- `save/iu_xray/qwen_dino_dpo_dino_top20_b4rouge_pairwise_lr1e6/checkpoints/checkpoint_epoch0_step383_train.pth`

最佳测试结果：

- `Bleu_4=0.134333`
- `ROUGE_L=0.369283`
- `CIDEr=0.282451`

问题：

- 明显低于 pseudo-label SFT。
- pairwise DPO 没有提升候选选择能力，反而破坏了已有生成分布。

结论：

- 当前这版 DPO 不应该推广。
- 仅靠 chosen/rejected 的文本偏好，不足以让模型学会图像条件下的细粒度选择。

## 10. Reranker / selector 尝试

由于 top20 oracle 很高，继续尝试可部署的 selector：不看参考答案，只根据图像、候选报告和模型自身信号选 top20 中的一条。

### 10.1 MLP pair reranker

方法：

- 输入：
  - query 图像 DINO embedding
  - candidate 图像 DINO embedding
  - 两者乘积、差值
  - 相似度、rank、候选报告长度、句子数、`xxxx` 比例、报告频次等 scalar feature
- 训练目标：
  - 在 train 样本 top20 候选中，预测哪个候选的 `B4 + ROUGE_L` 最高。

输出目录：

- `save/iu_xray/dino_reranker_b4rouge/`

结果：

- best val epoch:
  - `Bleu_4=0.129006`
  - `ROUGE_L=0.343123`
  - `B4+Rouge=0.472129`
- test:
  - `Bleu_4=0.117220`
  - `ROUGE_L=0.332974`
  - `B4+Rouge=0.450195`

问题：

- 虽然比 DINO top1 retrieval 好，但比生成模型差很多。
- val candidate hit 只有约 8%-9%，说明模型几乎没学会选 oracle candidate。

### 10.2 scalar / candidate-prior reranker

方法：

- 给每条候选报告增加 candidate prior：
  - 该报告在训练集中作为候选时的平均 reward
  - 最大 reward
  - 75 分位 reward
  - 出现频次
- 试了：
  - prior only
  - ridge regression
  - HistGradientBoosting
  - RandomForest

输出目录：

- `save/iu_xray/dino_scalar_reranker_b4rouge/`

结果：

- best val:
  - RandomForest: `Bleu_4=0.135396, ROUGE_L=0.354367`
- best test:
  - Ridge: `Bleu_4=0.129237, ROUGE_L=0.345514`

改进：

- 比 MLP 在 val 上更稳。
- candidate prior 确实提供了一些信号。

不足：

- 仍显著低于最佳 SFT 生成模型。
- 它更像是在学“哪条候选报告通常好”，不是在理解当前图像。

### 10.3 MRG model image-conditioned log-likelihood selector

方法：

- 对每个 val/test 样本的 DINO top20 候选报告，用当前最好 SFT 模型计算条件 log-likelihood。
- 选择模型认为 logprob 最高的候选。

使用 checkpoint：

- `save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`

输出目录：

- `save/iu_xray/dino_llm_logp_selector_sft_top20/`

val 结果：

- `Bleu_4=0.126382`
- `ROUGE_L=0.337050`
- `B4+Rouge=0.463432`

问题：

- 比 scalar reranker 差。
- 说明当前 MRG 模型虽然能生成通用报告，但还不是一个可靠的候选报告判别器。
- val 已经明显不佳，因此 test 提前停止，避免浪费 GPU 时间。

## 11. 总体问题分析

当前主要瓶颈不是单一训练参数，而是任务形态和模型行为：

1. 模型输出高度模板化
   - 最好模型的 unique outputs 仍很低。
   - 模型倾向输出常见正常报告。

2. 固定模板无法达标
   - 单模板 test B4 上界约 `0.151`。
   - 目标 `0.17` 要求更多图像/病例差异。

3. 检索候选池有潜力，但 selector 失败
   - top20 oracle test: `B4=0.288231, ROUGE_L=0.493809`
   - 但可部署 selector 最好 test 仍只有 `B4≈0.129, Rouge≈0.346`

4. 现有图像表征不足以直接排序报告
   - DINO top1 很差。
   - 简单 DINO embedding reranker 学不到 oracle 选择。

5. 当前生成模型也不会做候选判别
   - image-conditioned log-likelihood selector 表现差。
   - 说明模型 logprob 更偏语言先验，而不是图像-报告匹配。

## 12. 后续更可能有效的方向

### 12.1 检索增强生成，而不是 hard selection

目前 hard selection 的问题是：top20 中可能有多个候选各自包含部分正确描述，但单选一条容易错。

更合理的方法：

- 给模型输入 top-k 候选报告摘要或若干候选句子。
- 让模型在候选基础上融合生成最终报告。
- 训练时仍用 gold report 做 SFT。

可能形式：

```text
[image]
候选报告1: ...
候选报告2: ...
候选报告3: ...
请结合图像和候选报告生成最终报告。
```

风险：

- prompt 变长，Qwen3-8B 显存和速度压力会增大。
- 如果候选中错误内容多，模型可能复制错误。

### 12.2 句子级检索与融合

报告级候选太粗。可以把 train 报告拆成句子，检索 top-k 相似图像报告中的句子，再生成或选择句子集合。

优点：

- B4/Rouge 对 n-gram 和 LCS 敏感，句子级融合更容易提高重叠。
- 可以组合多个候选报告中的正确片段。

不足：

- 需要设计句子去重、排序和冲突消解。
- 可能生成不连贯或医学事实冲突的报告。

### 12.3 训练 cross-encoder reranker

当前 reranker 只用 DINO embedding 和简单文本统计，没有真正建模“图像和候选报告是否匹配”。

更强方案：

- 用当前 MRG 模型的 visual tokens + candidate report tokens 做 cross-encoder。
- 训练目标是 top20 candidate 的 B4/Rouge reward。
- 推理时对 top20 逐条打分。

优点：

- 比纯 DINO embedding 更可能理解候选报告文本。

不足：

- 训练和推理更慢。
- 需要小心避免过拟合 n-gram reward。

### 12.4 先提升视觉侧，再做 SFT/DPO

当前冻结 DINOv2 可能无法区分报告需要的细粒度异常。

可尝试：

- 解冻 resampler 已经在训练，但 vision encoder 一直 freeze。
- 尝试小学习率解冻 DINO 后几层。
- 或引入 CXR 专用视觉模型作为 encoder。

风险：

- 数据很小，解冻 vision encoder 容易过拟合。
- 训练成本上升。

### 12.5 不再优先做普通 DPO

已观察到多个 DPO 版本会退化：

- 原始 DPO 失败。
- gold/ref/worst 类 DPO 没有达到目标。
- DINO top20 pairwise DPO 也退化。

普通 DPO 暂时不是优先方向，除非 preference 数据和模型输入形式发生变化，例如基于检索增强输入构造 preference。

## 13. 当前建议

短期内不要继续盲目做：

- 更大 decode sweep
- 更大学习率 SFT
- 普通 pairwise DPO
- 只基于 DINO embedding 的 selector

更值得继续的是：

1. 检索增强生成：把 top-k 报告作为文本上下文输入生成模型。
2. 句子级检索融合：从 top-k 近邻报告中选句子，而不是选整条报告。
3. cross-encoder reranker：让模型同时看 image tokens 和 candidate report tokens 来打分。

当前最好模型仍可作为 baseline：

- `save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth`
- test: `Bleu_4=0.139392, ROUGE_L=0.375162`
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

### 6. MRG model conditional log-prob candidate selection

Method:
- Used the copy-oracle SFT checkpoint.
- At inference, scored each retrieved candidate report by conditional average token log-prob under the current MRG model with the same top-20 context.
- Selected the highest log-prob candidate instead of free-generating.

Result:
- Test: `B4=0.063173`, `Rouge-L=0.273543`, `CIDEr=0.078336`.
- Average selected rank: `9.69`; top-1 selected only `4.97%`.

Problem:
- The model's sequence likelihood is dominated by language prior and length/style preference, not candidate correctness.
- This approach is worse than free generation and much worse than the generic template.

### 7. Candidate-number SFT

Method:
- Added configurable `--retrieval_instruction` to the MRG model wrapper so retrieval prompts can ask for candidate selection instead of report generation.
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
- The current MRG model does not reliably use long top-k retrieved context for selection, even when the target is only a number.

Most likely next improvement:
- Replace DINO-only retrieval/reranking with a radiology image-text aligned retriever/reranker.
- Train a contrastive image-report model or cross-encoder on IU X-Ray plus larger MIMIC-CXR image-report pairs, then use it to rerank top-k candidates before generation/copy.
- For small IU X-Ray alone, a fixed template still remains the strongest non-oracle B4/Rouge baseline, but it cannot meet the new CheXbert target.
