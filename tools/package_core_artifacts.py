#!/usr/bin/env python3
import glob
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path("/data/wang.xuanmin/R2GenGPT")
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path(f"/data/wang.xuanmin/R2GenGPT_core_pack_{STAMP}")


EXACT_FILES = [
    "README.md",
    "UPGRADE_PLAN.md",
    "train.py",
    "configs/config.py",
    "dataset/data_helper.py",
    "models/R2GenGPT.py",
    "models/vision_resampler.py",
    "data/iu_xray/annotation.json",
    "data/iu_xray/annotation_rag_top20_copy_oracle_cw1.json",
    "data/iu_xray/annotation_rag_top20_number_cw1.json",
    "save/iu_xray/model_registry/README.txt",
    "save/iu_xray/model_registry/EXPERIMENT_LOG.md",
    "save/iu_xray/model_registry/B4_ROUGE_EXPERIMENT.md",
    "save/iu_xray/model_registry/IUXRAY_TRIAL_AND_ERROR_REVIEW.md",
    "save/iu_xray/model_registry/DPO_DIAGNOSIS.md",
    "save/iu_xray/model_registry/SFT_RESAMPLED_EXPERIMENT.md",
    "save/iu_xray/model_registry/TARGET_TEMPLATE_B4ROUGE.txt",
    "save/iu_xray/model_registry/clinical_metric_summary_20260609.json",
    "save/iu_xray/model_registry/RERANKER_EXPERIMENT_SUMMARY.json",
]


GLOB_PATTERNS = [
    "tools/*.py",
    "scripts/*.sh",
    "save/iu_xray/manual_template_target_b4rouge/**/*.json",
    "save/iu_xray/manual_template_target_b4rouge/**/*.txt",
    "save/iu_xray/template_refinement_b4rouge/**/*.json",
    "save/iu_xray/template_sentence_fusion_b4rouge_fast/**/*.json",
    "save/iu_xray/rag_top20_b4rouge/build_summary.json",
    "save/iu_xray/rag_top50_b4rouge/build_summary.json",
    "save/iu_xray/rag_top20_copy_oracle_cw1/**/*.json",
    "save/iu_xray/rag_top20_number_cw1/**/*.json",
    "save/iu_xray/rag_top50_clinical_oracle/*summary.json",
    "save/iu_xray/rag_top50_clinical_oracle/test_cw1.0_rf0.0_preds.json",
    "save/iu_xray/rag_top50_clinical_oracle/test_cw1.0_rf0.0_radgraph.json",
    "save/iu_xray/clinical_scalar_selector_top50/summary.json",
    "save/iu_xray/label_guided_selector_top50/summary.json",
    "save/iu_xray/text_cross_reranker_top50_cw0/summary.json",
    "save/iu_xray/text_cross_reranker_top50_cw0/*_metrics.json",
    "save/iu_xray/template_candidate_hybrid/summary.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5/log.txt",
    "save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5_test_beam3/run.log",
    "save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5_test_beam3/result/*.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5_test_beam3/clinical_metrics.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_copy_oracle_cw1_lr2e5_logp_top20/*.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_number_cw1_lr5e5/log.txt",
    "save/iu_xray/qwen_dino_sft_rag_top20_number_cw1_lr5e5_infer_beam1/*.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_number_cw1_lr5e5_infer_epoch0_beam1/*.json",
    "save/iu_xray/qwen_dino_sft_rag_top20_number_cw1_lr5e5_infer_epoch1_beam1/*.json",
]


EXCLUDE_SUFFIXES = {".pth", ".pt", ".bin", ".safetensors", ".ckpt", ".png", ".jpg", ".jpeg"}


def copy_file(rel: str, copied: list[str]) -> None:
    src = ROOT / rel
    if not src.exists() or not src.is_file():
        return
    if src.suffix in EXCLUDE_SUFFIXES:
        return
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel in EXACT_FILES:
        copy_file(rel, copied)
    for pattern in GLOB_PATTERNS:
        for src in glob.glob(str(ROOT / pattern), recursive=True):
            path = Path(src)
            if path.is_file():
                rel = str(path.relative_to(ROOT))
                copy_file(rel, copied)

    copied = sorted(set(copied))
    checkpoint_rows = []
    for path in sorted((ROOT / "save/iu_xray").glob("**/*.pth")):
        checkpoint_rows.append((str(path.relative_to(ROOT)), path.stat().st_size))

    manifest = {
        "source_root": str(ROOT),
        "pack_dir": str(OUT),
        "num_files": len(copied),
        "files": copied,
        "checkpoints_not_included": [
            {"path": path, "bytes": size} for path, size in checkpoint_rows
        ],
    }
    (OUT / "PACK_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    checkpoint_md = [
        "# Checkpoints Not Included",
        "",
        "Large model weight files were intentionally excluded from this GitHub-ready core pack.",
        "They remain on the remote server under the source project path.",
        "",
        "| Size | Path |",
        "| ---: | --- |",
    ]
    for path, size in checkpoint_rows:
        checkpoint_md.append(f"| {size} | `{path}` |")
    (OUT / "CHECKPOINTS_NOT_INCLUDED.md").write_text("\n".join(checkpoint_md) + "\n")

    readme = [
        "# R2GenGPT IU X-Ray Experiment Core Pack",
        "",
        "This pack contains the core code, scripts, experiment notes, annotations, summaries, metrics, and selected prediction JSON files for the IU X-Ray B4/Rouge/CheXbert/RadGraph optimization work.",
        "",
        "Large checkpoints are excluded. See `CHECKPOINTS_NOT_INCLUDED.md`.",
        "",
        "Important entry points:",
        "- `save/iu_xray/model_registry/EXPERIMENT_LOG.md`",
        "- `save/iu_xray/model_registry/B4_ROUGE_EXPERIMENT.md`",
        "- `save/iu_xray/model_registry/IUXRAY_TRIAL_AND_ERROR_REVIEW.md`",
        "- `tools/build_iuxray_copy_oracle_annotation.py`",
        "- `tools/run_iuxray_candidate_number_inference.py`",
        "- `tools/evaluate_iuxray_clinical_metrics.py`",
        "",
    ]
    (OUT / "README_CORE_PACK.md").write_text("\n".join(readme))

    print(str(OUT))
    print(f"copied {len(copied)} files")


if __name__ == "__main__":
    main()
