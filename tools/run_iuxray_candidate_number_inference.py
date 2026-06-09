#!/usr/bin/env python3
import argparse
import json
import os
import re
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
    cfg.max_length = 8
    cfg.precision = "bf16-mixed"
    cfg.savedmodel_path = args.output_dir
    cfg.num_workers = 0
    cfg.beam_size = args.beam_size
    cfg.min_new_tokens = args.min_new_tokens
    cfg.max_new_tokens = args.max_new_tokens
    cfg.no_repeat_ngram_size = 0
    cfg.repetition_penalty = 1.0
    cfg.length_penalty = 1.0
    cfg.temperature = 0
    cfg.retrieval_instruction = args.retrieval_instruction
    cfg.prompt = args.prompt
    return cfg


def parse_rank(text, topk):
    nums = [int(x) for x in re.findall(r"\d+", text)]
    for num in nums:
        if 1 <= num <= topk:
            return num - 1
    return 0


def write_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation_rag_top20_number_cw1.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--llm_model", default="resources/models/Qwen3-8B")
    parser.add_argument("--delta_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--retrieval_instruction", default="Select the single best candidate report for this chest X-ray image. Reply with only the candidate number from 1 to 20.")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ann = json.load(open(args.annotation))
    model_args = make_model_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R2GenGPT(model_args).to(device).eval()
    dataset = ParseDataset(model_args, split=args.split)
    ann_by_id = {str(item["id"]): item for item in ann[args.split]}

    refs = {}
    preds = {}
    raw_outputs = {}
    choices = {}
    for start in range(0, len(dataset), args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        samples = {
            "id": [str(item["id"]) for item in batch_items],
            "image": [[image.to(device) for image in item["image"]] for item in batch_items],
            "input_text": [item["input_text"] for item in batch_items],
        }
        if "retrieved_context" in batch_items[0]:
            samples["retrieved_context"] = [item.get("retrieved_context", "") for item in batch_items]
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model._generate_batch(samples)
            else:
                outputs = model._generate_batch(samples)
        for study_id, output in zip(samples["id"], outputs):
            item = ann_by_id[study_id]
            rank = parse_rank(output, args.topk)
            retrieved = item.get("retrieved_reports", [])[: args.topk]
            if rank >= len(retrieved):
                rank = 0
            refs[study_id] = clean_report(item["report"])
            preds[study_id] = clean_report(retrieved[rank]) if retrieved else ""
            raw_outputs[study_id] = output
            choices[study_id] = {"rank": rank, "raw_output": output, "text": preds[study_id]}
        print(f"generated {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)

    metrics = corpus_score(refs, preds)
    metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
    metrics["unique_outputs"] = len(set(preds.values()))
    metrics["avg_rank"] = sum(item["rank"] for item in choices.values()) / len(choices)
    metrics["top1_rate"] = sum(1 for item in choices.values() if item["rank"] == 0) / len(choices)
    write_json_report(os.path.join(args.output_dir, f"{args.split}_refs.json"), refs)
    write_json_report(os.path.join(args.output_dir, f"{args.split}_preds.json"), preds)
    with open(os.path.join(args.output_dir, f"{args.split}_raw_outputs.json"), "w") as f:
        json.dump(raw_outputs, f, indent=2)
    with open(os.path.join(args.output_dir, f"{args.split}_choices.json"), "w") as f:
        json.dump(choices, f, indent=2)
    with open(os.path.join(args.output_dir, f"{args.split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({args.split: metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
