# MRG_RL Experiment Artifact Pack

This directory tree is the curated artifact pack for our IU X-Ray MRG optimization work.

It contains the code, scripts, experiment notes, annotations, summaries, metrics, and selected prediction JSON files needed to inspect and reproduce the main trial sequence. Large checkpoints are excluded; see `CHECKPOINTS_NOT_INCLUDED.md`.

Primary reading order:

1. `README.md`
2. `save/iu_xray/model_registry/EXPERIMENT_LOG.md`
3. `save/iu_xray/model_registry/B4_ROUGE_EXPERIMENT.md`
4. `save/iu_xray/model_registry/IUXRAY_TRIAL_AND_ERROR_REVIEW.md`
5. `PACK_MANIFEST.json`

Key implemented tools:

- `tools/build_iuxray_rag_annotation.py`
- `tools/run_iuxray_clinical_oracle.py`
- `tools/evaluate_iuxray_clinical_metrics.py`
- `tools/train_iuxray_label_guided_selector.py`
- `tools/train_iuxray_text_cross_reranker.py`
- `tools/build_iuxray_copy_oracle_annotation.py`
- `tools/build_iuxray_candidate_number_annotation.py`
- `tools/run_iuxray_candidate_number_inference.py`

Compatibility note: some code files retain legacy class or path names from the remote experiment workspace. The repository should be read as our MRG/RL experiment package, not as an upstream framework mirror.
