# DPO Diagnosis

## Rollback
Promoted/current usable checkpoint is the SFT continue2 checkpoint:

`save/iu_xray/qwen_dino_sft_8gpu_continue2/checkpoints/checkpoint_epoch1_step190_train.pth`

The DPO checkpoint is retained for inspection but should not be used as the current best model:

`save/iu_xray/qwen_dino_dpo_8gpu/checkpoints/checkpoint_epoch0_step366_train.pth`

This failed checkpoint has since been deleted; test result files are retained under `save/iu_xray/qwen_dino_dpo_8gpu_test/result/`.

## Observed Failure
DPO test generation collapsed to repeated `xxxx` tokens.

DPO test metrics:
- Bleu_1: 0.00710078534031402
- Bleu_2: 0.0021624842050101786
- Bleu_3: 0.0005371206027691384
- Bleu_4: 4.0092496930821405e-08
- ROUGE_L: 0.011297340787916504
- CIDEr: 3.930043530942576e-05

Best SFT continue2 test metrics before DPO:
- Bleu_4: 0.138284
- CIDEr: 0.282538

## Checks
- DPO training completed with exit code 0.
- DPO saved `checkpoint_epoch0_step366_train.pth`.
- No runtime crash was found.
- Preference file has 2924 pairs.
- Top chosen report appears 1234 times; top 4 chosen reports dominate the dataset.
- DPO loss in `models/R2GenGPT.py` is reference-free: `-logsigmoid(beta * (policy_chosen_logp - policy_rejected_logp))`.
- There is no frozen SFT/reference log-ratio term and no SFT/NLL regularizer.

## Likely Causes
1. Preference data is dominated by a few generic chosen templates.
2. The implemented objective is not full DPO; it lacks a reference-model anchor/KL behavior.
3. Chosen reports are much shorter than rejected reports, and `dpo_average_logps=True` can amplify length/template bias.

## Non-destructive Mitigation Artifact
Created a filtered preference file for a safer next experiment:

`data/iu_xray/preferences_cap50_margin010.jsonl`

It keeps 726 pairs, requires reward margin >= 0.10, and caps each exact chosen text at 50 occurrences.

## Suggested Next Experiment
Start from the SFT continue2 checkpoint, use the filtered preference file, lower LR to around 1e-6, and either add a reference-logit term or add an SFT/NLL regularizer. Do not promote a DPO checkpoint until test generation is inspected.

## Implemented Fix Attempt: Reference-Anchored DPO
Code now supports a safer DPO objective:
- `--dpo_objective reference` subtracts frozen SFT/reference log-prob margins from policy margins.
- `--dpo_sft_loss_weight` adds an optional SFT/NLL anchor during DPO.
- `--dpo_require_ref_logps` enforces precomputed reference logps in preference rows.
- `precompute_reference_logps.py` writes `ref_chosen_logp` and `ref_rejected_logp` into JSONL preferences.

Experiments completed:
1. `qwen_dino_dpo_safe_ref_cap50_margin010`, using capped reward preferences.
   - Checkpoint: `save/iu_xray/qwen_dino_dpo_safe_ref_cap50_margin010/checkpoints/checkpoint_epoch0_step91_train.pth`
   - Test: Bleu_4=0.1350262714, ROUGE_L=0.3749023117, CIDEr=0.2626901877
   - Not promoted: lower CIDEr than SFT and output still mostly one template.

2. `qwen_dino_dpo_gold_ref_vs_worst`, using gold train reports as chosen and worst generated candidate as rejected.
   - Preference file: `data/iu_xray/preferences_gold_ref_vs_worst_ref.jsonl` (3062 rows)
   - Checkpoint: `save/iu_xray/qwen_dino_dpo_gold_ref_vs_worst/checkpoints/checkpoint_epoch0_step383_train.pth`
   - Test: Bleu_4=0.1324304566, ROUGE_L=0.3636438662, CIDEr=0.3059699283
   - Not promoted under strict rule: CIDEr/ROUGE_L improved, but Bleu_4 dropped below SFT and generated text had higher `xxxx`/template rate.

3. `qwen_dino_dpo_gold_no_xxxx_conservative`, filtering chosen reports containing `xxxx`, with lower LR/beta and stronger SFT anchor.
   - Preference file: `data/iu_xray/preferences_gold_ref_no_xxxx_vs_worst_ref.jsonl` (1637 rows)
   - Checkpoint: `save/iu_xray/qwen_dino_dpo_gold_no_xxxx_conservative/checkpoints/checkpoint_epoch0_step205_train.pth`
   - Test: Bleu_4=0.1255175369, ROUGE_L=0.3547481518, CIDEr=0.2502487154
   - Not promoted: worse than SFT and worse than the gold_ref_vs_worst DPO candidate.

## Current Recommendation
Keep `qwen_dino_sft_8gpu_continue2` as the promoted model. The DPO implementation is no longer catastrophically broken, but the preference construction still pushes toward generic report templates. To exceed SFT reliably, the next improvement should focus on preference data quality: image-conditioned negatives and positives, stronger diversity constraints, and filtering/normalizing placeholder artifacts before reward construction rather than only changing beta/LR.

