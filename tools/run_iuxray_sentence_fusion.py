#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.train_iuxray_dino_reranker import (
    build_groups,
    clean_report,
    encode_or_load,
)
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge


def split_sentences(report):
    report = clean_report(report)
    sentences = []
    for sent in report.split(" . "):
        sent = sent.strip()
        if sent == ".":
            continue
        sent = sent[:-2].strip() if sent.endswith(" .") else sent
        if sent:
            sentences.append(sent)
    return sentences


def join_sentences(sentences):
    if not sentences:
        return "no acute cardiopulmonary abnormality ."
    return " . ".join(sentences) + " ."


def b4_rouge_score(refs, hypos):
    ref = {key: [refs[key]] for key in refs}
    hyp = {key: [hypos[key]] for key in refs}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return {"Bleu_4": float(bleu[3]), "ROUGE_L": float(rouge)}


def build_sentence_priors(train_reports):
    freq = Counter()
    report_count = Counter()
    for report in train_reports.values():
        seen = set()
        for sent in split_sentences(report):
            freq[sent] += 1
            seen.add(sent)
        for sent in seen:
            report_count[sent] += 1
    return freq, report_count


def fuse_one(candidate_reports, sent_freq, cfg):
    pool = {}
    for rank, report in enumerate(candidate_reports):
        for pos, sent in enumerate(split_sentences(report)):
            words = sent.split()
            if len(words) < cfg["min_words"]:
                continue
            freq_score = math.log1p(sent_freq.get(sent, 0))
            rank_score = 1.0 / (rank + 1)
            pos_score = 1.0 / (pos + 1)
            len_score = -abs(len(words) - cfg["target_words"]) / cfg["target_words"]
            score = (
                cfg["w_freq"] * freq_score
                + cfg["w_rank"] * rank_score
                + cfg["w_pos"] * pos_score
                + cfg["w_len"] * len_score
            )
            if sent not in pool or score > pool[sent]["score"]:
                pool[sent] = {"score": score, "rank": rank, "pos": pos, "sent": sent}
    selected = sorted(pool.values(), key=lambda item: item["score"], reverse=True)[: cfg["num_sent"]]
    # Keep report readable by preserving retrieval/report order after selection.
    selected = sorted(selected, key=lambda item: (item["rank"], item["pos"]))
    return join_sentences([item["sent"] for item in selected])


def make_predictions(group, train_raw_reports, sent_freq, cfg):
    preds = {}
    for study_id, candidate_ids in zip(group["ids"], group["candidate_ids"]):
        reports = [train_raw_reports[cand_id] for cand_id in candidate_ids[: cfg["topk"]]]
        preds[study_id] = fuse_one(reports, sent_freq, cfg)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--output_dir", default="save/iu_xray/sentence_fusion_b4rouge")
    parser.add_argument("--max_topk", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    raw_reports = {split: {str(item["id"]): item["report"] for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_reports = reports["train"]
    train_raw_reports = raw_reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = Counter(train_reports.values())
    sent_freq, _ = build_sentence_priors(train_raw_reports)

    groups = {}
    for split in ["val", "test"]:
        groups[split] = build_groups(
            split,
            ids[split],
            reports,
            embeddings,
            train_ids,
            train_matrix,
            train_reports,
            report_freq,
            args.max_topk,
            with_labels=False,
        )

    grid = []
    for topk, num_sent, w_freq, w_rank, w_pos, w_len, target_words in itertools.product(
        [3, 5, 10],
        [5, 6],
        [0.5, 1.0, 1.5],
        [0.2, 0.8],
        [0.1, 0.5],
        [0.0, 0.2],
        [8],
    ):
        grid.append({
            "topk": topk,
            "num_sent": num_sent,
            "w_freq": w_freq,
            "w_rank": w_rank,
            "w_pos": w_pos,
            "w_len": w_len,
            "target_words": target_words,
            "min_words": 3,
        })

    rows = []
    for idx, cfg in enumerate(grid, start=1):
        preds = make_predictions(groups["val"], train_raw_reports, sent_freq, cfg)
        metrics = b4_rouge_score(reports["val"], preds)
        row = {"score": metrics["Bleu_4"] + metrics["ROUGE_L"], "metrics": metrics, "cfg": cfg}
        rows.append(row)
        if idx % 20 == 0:
            print(f"searched {idx}/{len(grid)} best={max(r['score'] for r in rows):.6f}", flush=True)

    rows.sort(key=lambda row: row["score"], reverse=True)
    best = rows[0]
    final = {"best_val": best, "top_val": rows[:20]}

    for split in ["val", "test"]:
        preds = make_predictions(groups[split], train_raw_reports, sent_freq, best["cfg"])
        metrics = b4_rouge_score(reports[split], preds)
        metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
        metrics["unique_outputs"] = len(set(preds.values()))
        final[split] = metrics
        with open(os.path.join(args.output_dir, f"{split}_preds.json"), "w") as f:
            json.dump({k: [v] for k, v in preds.items()}, f, indent=2)
        with open(os.path.join(args.output_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
