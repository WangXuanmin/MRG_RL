#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from tools.run_iuxray_template_sentence_fusion import BASE_IDS, join_sentences, split_sentences
from tools.train_iuxray_dino_reranker import clean_report


def b4_rouge(refs, text):
    ref = {k: [v] for k, v in refs.items()}
    hyp = {k: [text] for k in refs}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return {"Bleu_4": float(bleu[3]), "ROUGE_L": float(rouge), "B4_Rouge": float(bleu[3] + rouge)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--output_dir", default="save/iu_xray/template_global_sentence_b4rouge")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ann = json.load(open(args.annotation))
    reports = {sp: {str(x["id"]): clean_report(x["report"]) for x in ann[sp]} for sp in ["train", "val", "test"]}
    raw_train = {str(x["id"]): x["report"] for x in ann["train"]}

    sent_freq = {}
    for item in ann["train"]:
        for sent in split_sentences(item["report"]):
            words = sent.split()
            if 4 <= len(words) <= 18:
                sent_freq[sent] = sent_freq.get(sent, 0) + 1
    candidate_sentences = [
        sent
        for sent, _ in sorted(sent_freq.items(), key=lambda item: item[1], reverse=True)[:200]
    ]

    rows = []
    for base_id in BASE_IDS:
        if base_id not in raw_train:
            continue
        base_sents = split_sentences(raw_train[base_id])
        variants = [("none", join_sentences(base_sents))]
        for sent in candidate_sentences:
            if sent in base_sents:
                continue
            variants.append((sent, join_sentences(base_sents[:-1] + [sent] + base_sents[-1:])))
        for sent, text in variants:
            m = b4_rouge(reports["val"], text)
            rows.append({"score": m["B4_Rouge"], "metrics": m, "base_id": base_id, "extra_sentence": sent, "text": text})
        print(f"base {base_id} done variants={len(variants)} best={max(row['score'] for row in rows):.6f}", flush=True)

    rows.sort(key=lambda x: x["score"], reverse=True)
    final = {"top_val": rows[:50]}
    for idx, row in enumerate(rows[:50], 1):
        m = b4_rouge(reports["test"], row["text"])
        row["test_metrics"] = m
        if idx <= 10:
            print(idx, row["base_id"], row["extra_sentence"], row["metrics"], m, flush=True)

    best = rows[0]
    final["best_val"] = best
    final["test_for_best_val"] = best["test_metrics"]
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(args.output_dir, "best_val_pred.json"), "w") as f:
        json.dump({sid: [best["text"]] for sid in reports["test"]}, f, indent=2)
    print("FINAL")
    print(json.dumps({"best_val": {k: v for k, v in best.items() if k != "text"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
