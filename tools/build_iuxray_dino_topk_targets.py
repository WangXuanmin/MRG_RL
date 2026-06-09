#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge


def clean_report(report):
    report_cleaner = lambda t: t.replace("..", ".").replace("..", ".").replace("..", ".").replace("1. ", "") \
        .replace(". 2. ", ". ").replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ") \
        .replace(" 2. ", ". ").replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ") \
        .strip().lower().split(". ")
    sent_cleaner = lambda t: re.sub(r"[.,?;*!%^&_+():\-\[\]{}]", "", t.replace("\"", "").replace("/", "")
                                .replace("\\", "").replace("'", "").strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    return " . ".join(tokens) + " ." if tokens else ""


def pair_score(ref_text, hyp_text):
    ref = {"x": [ref_text]}
    hyp = {"x": [hyp_text]}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return float(bleu[3] + rouge), float(bleu[3]), float(rouge)


def encode_studies(annotation, base_dir, vision_model, batch_size):
    flat = []
    for split_name, split_items in annotation.items():
        for item in split_items:
            for image_path in item["image_path"]:
                flat.append((split_name, str(item["id"]), os.path.join(base_dir, image_path)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(vision_model)
    model = AutoModel.from_pretrained(vision_model).to(device).eval()
    sums = defaultdict(lambda: None)
    counts = defaultdict(int)

    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch = flat[start:start + batch_size]
            images = []
            for _, _, path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images, return_tensors="pt").to(device)
            embeds = model(**inputs).last_hidden_state.mean(dim=1)
            embeds = F.normalize(embeds.float(), dim=1).cpu()
            for (split_name, study_id, _), embed in zip(batch, embeds):
                key = (split_name, study_id)
                sums[key] = embed.clone() if sums[key] is None else sums[key] + embed
                counts[key] += 1
            if (start // batch_size + 1) % 20 == 0:
                print(f"encoded {start + len(batch)}/{len(flat)}", flush=True)
    return {key: F.normalize((value / counts[key]).unsqueeze(0), dim=1).squeeze(0) for key, value in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--output_annotation", required=True)
    parser.add_argument("--output_preferences", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    annotation = json.load(open(args.annotation))
    embeds = encode_studies(annotation, args.base_dir, args.vision_model, args.batch_size)
    train_items = annotation["train"]
    train_ids = [str(item["id"]) for item in train_items]
    train_reports = {str(item["id"]): clean_report(item["report"]) for item in train_items}
    train_raw_reports = {str(item["id"]): item["report"] for item in train_items}
    train_matrix = torch.stack([embeds[("train", study_id)] for study_id in train_ids])

    new_annotation = json.loads(json.dumps(annotation))
    preferences = []
    report_rows = []

    for idx, item in enumerate(new_annotation["train"]):
        study_id = str(item["id"])
        query = embeds[("train", study_id)]
        sims = train_matrix @ query
        order = sims.topk(min(args.topk + 1, len(train_ids))).indices.tolist()
        candidates = []
        for rank in order:
            cand_id = train_ids[rank]
            if cand_id == study_id:
                continue
            total, b4, rouge = pair_score(train_reports[study_id], train_reports[cand_id])
            candidates.append((total, b4, rouge, cand_id, float(sims[rank])))
            if len(candidates) >= args.topk:
                break
        candidates.sort(reverse=True, key=lambda row: row[0])
        chosen = candidates[0]
        rejected = candidates[-1]
        item["report"] = train_raw_reports[chosen[3]]
        preferences.append({
            "id": study_id,
            "chosen": train_raw_reports[chosen[3]],
            "rejected": train_raw_reports[rejected[3]],
            "chosen_reward": chosen[0],
            "rejected_reward": rejected[0],
            "chosen_source": f"dino_top{args.topk}:{chosen[3]}",
            "rejected_source": f"dino_top{args.topk}:{rejected[3]}",
            "chosen_bleu4": chosen[1],
            "chosen_rouge_l": chosen[2],
            "rejected_bleu4": rejected[1],
            "rejected_rouge_l": rejected[2],
            "chosen_similarity": chosen[4],
            "rejected_similarity": rejected[4],
        })
        report_rows.append(preferences[-1])
        if (idx + 1) % 500 == 0:
            print(f"selected {idx + 1}/{len(new_annotation['train'])}", flush=True)

    os.makedirs(os.path.dirname(args.output_annotation) or ".", exist_ok=True)
    with open(args.output_annotation, "w") as f:
        json.dump(new_annotation, f, indent=2)
    with open(args.output_preferences, "w") as f:
        for row in preferences:
            f.write(json.dumps(row) + "\n")

    avg_chosen = sum(row["chosen_reward"] for row in report_rows) / len(report_rows)
    avg_rejected = sum(row["rejected_reward"] for row in report_rows) / len(report_rows)
    summary = {
        "num_pairs": len(preferences),
        "topk": args.topk,
        "avg_chosen_reward": avg_chosen,
        "avg_rejected_reward": avg_rejected,
        "avg_margin": avg_chosen - avg_rejected,
        "first_rows": report_rows[:5],
    }
    with open(args.output_report, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
