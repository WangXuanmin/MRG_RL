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
from tools.train_iuxray_dino_reranker import clean_report, corpus_score


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
    cfg.max_length = args.max_length
    cfg.precision = "bf16-mixed"
    cfg.dpo_average_logps = True
    cfg.savedmodel_path = args.output_dir
    cfg.num_workers = 0
    cfg.prompt = args.prompt
    return cfg


def write_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def score_split(model, dataset, ann_items, topk, chunk_size, device):
    refs = {}
    preds = {}
    choices = {}
    dataset_by_id = {str(dataset[i]["id"]): dataset[i] for i in range(len(dataset))}
    for idx, item in enumerate(ann_items):
        study_id = str(item["id"])
        sample = dataset_by_id[study_id]
        sample_images = [image.to(device) for image in sample["image"]]
        retrieved_context = sample.get("retrieved_context", None)
        candidates = [clean_report(text) for text in item.get("retrieved_reports", [])[:topk]]
        candidates = [text for text in candidates if text]
        if not candidates:
            candidates = [clean_report(item["report"])]
        scores = []
        for start in range(0, len(candidates), chunk_size):
            reports = candidates[start : start + chunk_size]
            batch = {
                "image": [sample_images for _ in reports],
                "retrieved_context": [retrieved_context for _ in reports],
            }
            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        row_scores = model._sequence_logps(batch, reports)
                else:
                    row_scores = model._sequence_logps(batch, reports)
            scores.extend([float(value) for value in row_scores.detach().cpu()])
        best = max(range(len(candidates)), key=lambda i: scores[i])
        refs[study_id] = clean_report(item["report"])
        preds[study_id] = candidates[best]
        choices[study_id] = {
            "rank": int(best),
            "score": float(scores[best]),
            "scores": scores,
            "text": candidates[best],
        }
        if (idx + 1) % 25 == 0:
            print(f"scored {idx + 1}/{len(ann_items)}", flush=True)
    return refs, preds, choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation_rag_top20_copy_oracle_cw1.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--llm_model", default="resources/models/Qwen3-8B")
    parser.add_argument("--delta_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--prompt", default="Generate a comprehensive and detailed diagnosis report for this chest X-ray image.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ann = json.load(open(args.annotation))
    model_args = make_model_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R2GenGPT(model_args).to(device).eval()
    summary = {}

    for split in args.splits:
        dataset = ParseDataset(model_args, split=split)
        refs, preds, choices = score_split(model, dataset, ann[split], args.topk, args.chunk_size, device)
        metrics = corpus_score(refs, preds)
        metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
        metrics["unique_outputs"] = len(set(preds.values()))
        metrics["avg_rank"] = sum(item["rank"] for item in choices.values()) / len(choices)
        metrics["top1_rate"] = sum(1 for item in choices.values() if item["rank"] == 0) / len(choices)
        summary[split] = metrics
        write_json_report(os.path.join(args.output_dir, f"{split}_preds.json"), preds)
        write_json_report(os.path.join(args.output_dir, f"{split}_refs.json"), refs)
        with open(os.path.join(args.output_dir, f"{split}_choices.json"), "w") as f:
            json.dump(choices, f, indent=2)
        with open(os.path.join(args.output_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(split, json.dumps(metrics, indent=2), flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
