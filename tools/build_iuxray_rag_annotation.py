#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.train_iuxray_dino_reranker import (
    build_groups,
    clean_report,
    corpus_score,
    encode_or_load,
    evaluate_selector,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--output_annotation", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_raw_reports = {str(item["id"]): item["report"] for item in annotation["train"]}
    train_reports = reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = {}
    for text in train_reports.values():
        report_freq[text] = report_freq.get(text, 0) + 1

    new_annotation = json.loads(json.dumps(annotation))
    summary = {"topk": args.topk, "splits": {}}

    for split in ["train", "val", "test"]:
        group = build_groups(
            split,
            ids[split],
            reports,
            embeddings,
            train_ids,
            train_matrix,
            train_reports,
            report_freq,
            args.topk,
            with_labels=True,
        )
        id_to_candidates = {
            study_id: [train_raw_reports[cand_id] for cand_id in cand_ids]
            for study_id, cand_ids in zip(group["ids"], group["candidate_ids"])
        }
        for item in new_annotation[split]:
            item["retrieved_reports"] = id_to_candidates[str(item["id"])]

        top1_scores = torch.zeros(group["labels"].shape)
        top1_scores[:, 0] = 1.0
        oracle_scores = group["labels"]
        out_dir = os.path.dirname(args.output_summary) or "."
        refs = reports[split]
        summary["splits"][split] = {
            "num_items": len(group["ids"]),
            "top1": evaluate_selector(f"{split}_rag_build_top1", group, refs, train_reports, top1_scores, out_dir),
            "oracle": evaluate_selector(f"{split}_rag_build_oracle_top{args.topk}", group, refs, train_reports, oracle_scores, out_dir),
        }

    os.makedirs(os.path.dirname(args.output_annotation) or ".", exist_ok=True)
    with open(args.output_annotation, "w") as f:
        json.dump(new_annotation, f, indent=2)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
