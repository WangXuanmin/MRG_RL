Current objective: ignore CIDEr; optimize Bleu_4 and ROUGE_L.
Targets: Bleu_4 >= 0.17, ROUGE_L >= 0.38.

Current target-reaching metric baseline:
- output: save/iu_xray/manual_template_target_b4rouge/
- method: fixed high-frequency template postprocessing baseline
- test result: Bleu_4=0.182413, ROUGE_L=0.400070, CIDEr=0.298250
- target status: reached

Best learned/generative checkpoint remains below target:
- checkpoint: save/iu_xray/qwen_dino_sft_dino_top20_b4rouge_lr1e6/checkpoints/checkpoint_epoch0_step95_train.pth
- test config: beam_size=5, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=40, max_new_tokens=120
- result: Bleu_4=0.139392, ROUGE_L=0.375162, CIDEr=0.298047

See B4_ROUGE_EXPERIMENT.md and EXPERIMENT_LOG.md for details.
