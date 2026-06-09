# SFT Resampled Experiment

## Change
Generated `data/iu_xray/annotation_resampled_abnormal_cap.json` from the original IU Xray annotation:
- source train rows: 3062
- output train rows: 5615
- exact cleaned report cap: 30
- abnormal report repeat: 2
- abnormal terms: opacity, effusion, atelectasis, cardiomegaly, edema, consolidation, pneumothorax, infiltrate, nodule

Trained one SFT epoch from the previous promoted SFT checkpoint with lr=1e-6:
`save/iu_xray/qwen_dino_sft_resampled_abnormal_lr1e6/checkpoints/checkpoint_epoch0_step175_train.pth`

## Best Inference Config
`beam_size=5 length_penalty=1.0 repetition_penalty=1.0 no_repeat_ngram_size=2 min_new_tokens=40 max_new_tokens=120`

Result dir:
`save/iu_xray/resampled_decode_refine/beam5_lp1p0_rp1p0_ng2_min40_max120`

Metrics:
- Bleu_1: 0.3984365382
- Bleu_2: 0.2497106768
- Bleu_3: 0.1725590357
- Bleu_4: 0.1224755445
- ROUGE_L: 0.3342318271
- CIDEr: 0.3585407578
- Combined: 0.2405081512
- unique_outputs: 6
- xxxx_mean: 0.0

## Interpretation
This improves the selected combined score mainly through CIDEr while eliminating `xxxx` in the promoted decoding. The model is still template-heavy, but it is better than the prior SFT baseline and safer than DPO candidates under the agreed combined-score criterion.
