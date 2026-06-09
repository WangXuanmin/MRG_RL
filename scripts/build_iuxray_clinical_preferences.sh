#!/bin/bash

python -u build_clinical_preferences.py \
  --refs save/iu_xray/candidates/train_refs.json \
  --candidates \
    save/iu_xray/candidates/train_beam.json \
    save/iu_xray/candidates/train_sample_t07.json \
    save/iu_xray/candidates/train_sample_t10.json \
  --output data/iu_xray/preferences.jsonl \
  --rewards_output data/iu_xray/candidate_rewards.json \
  --work_dir save/iu_xray/clinical_pref_work \
  --min_margin 0.03 \
  --radgraph_weight 0.45 \
  --chexbert_weight 0.45 \
  --ngram_weight 0.10
