#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.config import parser as config_parser
from dataset.data_helper import ParseDataset
from models.R2GenGPT import R2GenGPT
from tools.train_iuxray_dino_reranker import (
    build_groups,
    clean_report,
    corpus_score,
    encode_or_load,
    evaluate_selector,
)


def make_model_args(args):
    cfg = config_parser.parse_args([])
    cfg.dataset = "iu_xray"
    cfg.annotation = args.annotation
    cfg.base_dir = args.base_dir
    cfg.vision_model = args.vision_model
    cfg.llm_model = args.llm_model
    cfg.llm_use_lora = True
    cfg.llm_r = 16
    cfg.llm_alpha = 32
    cfg.resampler_num_queries = 32
    cfg.resampler_num_layers = 2
    cfg.delta_file = args.delta_file
    cfg.freeze_vm = True
    cfg.max_length = 160
    cfg.precision = "bf16-mixed"
    cfg.dpo_average_logps = True
    cfg.savedmodel_path = args.output_dir
    cfg.num_workers = 0
    return cfg


def score_candidates(model, dataset_by_id, group, train_reports, chunk_size, device):
    all_avg_scores = []
    for row_idx, study_id in enumerate(group["ids"]):
        sample = dataset_by_id[study_id]
        sample_images = [image.to(device) for image in sample["image"]]
        cand_reports = [train_reports[cand_id] for cand_id in group["candidate_ids"][row_idx]]
        row_scores = []
        for start in range(0, len(cand_reports), chunk_size):
            reports = cand_reports[start:start + chunk_size]
            batch = {
                "image": [sample_images for _ in reports],
                "input_text": reports,
            }
            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        scores = model._sequence_logps(batch, reports)
                else:
                    scores = model._sequence_logps(batch, reports)
            row_scores.extend([float(x) for x in scores.detach().cpu()])
        all_avg_scores.append(row_scores)
        if (row_idx + 1) % 50 == 0:
            print(f"scored {row_idx + 1}/{len(group['ids'])}", flush=True)
    return torch.tensor(all_avg_scores, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--llm_model", default="resources/models/Qwen3-8B")
    parser.add_argument("--delta_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_reports = reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = {}
    for text in train_reports.values():
        report_freq[text] = report_freq.get(text, 0) + 1

    groups = {}
    for split in ["val", "test"]:
        groups[split] = build_groups(split, ids[split], reports, embeddings, train_ids, train_matrix, train_reports, report_freq, args.topk, with_labels=True)
        print("group", split, groups[split]["labels"].shape, flush=True)

    model_args = make_model_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R2GenGPT(model_args).to(device).eval()

    final = {}
    for split in ["val", "test"]:
        dataset = ParseDataset(model_args, split=split)
        dataset_by_id = {dataset[i]["id"]: dataset[i] for i in range(len(dataset))}
        scores = score_candidates(model, dataset_by_id, groups[split], train_reports, args.chunk_size, device)
        final[f"{split}_llm_logp"] = evaluate_selector(f"{split}_llm_logp", groups[split], reports[split], train_reports, scores, args.output_dir)
        final[f"{split}_llm_logp"]["candidate_hit"] = float((scores.argmax(dim=1) == groups[split]["labels"].argmax(dim=1)).float().mean())
        oracle = groups[split]["labels"]
        final[f"{split}_oracle_top{args.topk}"] = evaluate_selector(f"{split}_oracle_top{args.topk}", groups[split], reports[split], train_reports, oracle, args.output_dir)
        print(split, final[f"{split}_llm_logp"], flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
