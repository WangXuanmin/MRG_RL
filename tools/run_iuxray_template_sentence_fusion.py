#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from tools.train_iuxray_dino_reranker import build_groups, clean_report, encode_or_load


BASE_IDS = ["2792", "3576", "1429", "3258", "89", "114", "345", "90", "3304", "1437"]


def split_sentences(report):
    report = clean_report(report)
    out = []
    for sent in report.split(" . "):
        sent = sent.strip()
        if sent.endswith(" ."):
            sent = sent[:-2].strip()
        if sent and sent != ".":
            out.append(sent)
    return out


def join_sentences(sentences):
    dedup = []
    seen = set()
    for sent in sentences:
        if sent not in seen:
            seen.add(sent)
            dedup.append(sent)
    return " . ".join(dedup) + " ."


def b4_rouge(refs, hypos):
    ref = {k: [refs[k]] for k in refs}
    hyp = {k: [hypos[k]] for k in refs}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return {"Bleu_4": float(bleu[3]), "ROUGE_L": float(rouge), "B4_Rouge": float(bleu[3] + rouge)}


def sentence_freq(train_raw_reports):
    freq = Counter()
    for report in train_raw_reports.values():
        freq.update(split_sentences(report))
    return freq


def extra_sentences(candidate_reports, freq, cfg, base_sents):
    pool = {}
    base_set = set(base_sents)
    for rank, report in enumerate(candidate_reports[: cfg["topk"]]):
        for pos, sent in enumerate(split_sentences(report)):
            if sent in base_set:
                continue
            words = sent.split()
            if len(words) < cfg["min_words"] or len(words) > cfg["max_words"]:
                continue
            score = (
                cfg["w_freq"] * math.log1p(freq.get(sent, 0))
                + cfg["w_rank"] / (rank + 1)
                + cfg["w_pos"] / (pos + 1)
            )
            if sent not in pool or score > pool[sent]["score"]:
                pool[sent] = {"sent": sent, "score": score, "rank": rank, "pos": pos}
    selected = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[: cfg["num_extra"]]
    return [x["sent"] for x in sorted(selected, key=lambda x: (x["rank"], x["pos"]))]


def predict(group, train_raw_reports, base_report, freq, cfg):
    base_sents = split_sentences(base_report)
    preds = {}
    for sid, cand_ids in zip(group["ids"], group["candidate_ids"]):
        cands = [train_raw_reports[cid] for cid in cand_ids]
        extras = extra_sentences(cands, freq, cfg, base_sents)
        if cfg["placement"] == "before_impression":
            sents = base_sents[:-1] + extras + base_sents[-1:]
        else:
            sents = base_sents + extras
        preds[sid] = join_sentences(sents)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--output_dir", default="save/iu_xray/template_sentence_fusion_b4rouge")
    parser.add_argument("--max_topk", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ann = json.load(open(args.annotation))
    embeddings = encode_or_load(ann, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {sp: {str(x["id"]): clean_report(x["report"]) for x in ann[sp]} for sp in ["train", "val", "test"]}
    raw = {sp: {str(x["id"]): x["report"] for x in ann[sp]} for sp in ["train", "val", "test"]}
    ids = {sp: [str(x["id"]) for x in ann[sp]] for sp in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_matrix = torch.stack([embeddings[("train", sid)] for sid in train_ids])
    report_freq = Counter(reports["train"].values())
    freq = sentence_freq(raw["train"])

    groups = {}
    for split in ["val", "test"]:
        groups[split] = build_groups(split, ids[split], reports, embeddings, train_ids, train_matrix, reports["train"], report_freq, args.max_topk, with_labels=False)

    base_reports = {sid: raw["train"][sid] for sid in BASE_IDS if sid in raw["train"]}
    grid = []
    for base_id, topk, num_extra, w_freq, w_rank, w_pos, placement in itertools.product(
        list(base_reports),
        [3, 5],
        [0, 1],
        [0.8],
        [0.8],
        [0.1],
        ["before_impression"],
    ):
        grid.append({
            "base_id": base_id,
            "topk": topk,
            "num_extra": num_extra,
            "w_freq": w_freq,
            "w_rank": w_rank,
            "w_pos": w_pos,
            "placement": placement,
            "min_words": 4,
            "max_words": 16,
        })

    rows = []
    for i, cfg in enumerate(grid, 1):
        preds = predict(groups["val"], raw["train"], base_reports[cfg["base_id"]], freq, cfg)
        metrics = b4_rouge(reports["val"], preds)
        rows.append({"score": metrics["B4_Rouge"], "metrics": metrics, "cfg": cfg})
        if i % 10 == 0:
            print(f"searched {i}/{len(grid)} best={max(x['score'] for x in rows):.6f}", flush=True)
    rows.sort(key=lambda x: x["score"], reverse=True)
    best = rows[0]
    final = {"best_val": best, "top_val": rows[:20]}
    for split in ["val", "test"]:
        preds = predict(groups[split], raw["train"], base_reports[best["cfg"]["base_id"]], freq, best["cfg"])
        metrics = b4_rouge(reports[split], preds)
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
