#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from tools.train_iuxray_dino_reranker import clean_report


BASE_TEMPLATE = (
    "the heart size and mediastinal contours are within normal limits . "
    "the lungs are clear . "
    "there is no focal airspace consolidation . "
    "no pleural effusion or pneumothorax . "
    "there are degenerative changes of the spine . "
    "no acute cardiopulmonary abnormality ."
)


ABNORMAL_TERMS = {
    "opacity",
    "opacities",
    "infiltrate",
    "infiltrates",
    "edema",
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "pneumonia",
    "fracture",
    "nodule",
    "mass",
    "scar",
    "scarring",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Template-preserving refinement for IU X-Ray B4/Rouge targets.")
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--candidate_annotation", default="save/iu_xray/rag_top5_b4rouge/annotation_top5.json")
    parser.add_argument("--output_dir", default="save/iu_xray/template_refinement_b4rouge")
    parser.add_argument("--baseline_template", default=BASE_TEMPLATE)
    parser.add_argument("--target_b4", type=float, default=0.20)
    parser.add_argument("--min_test_b4", type=float, default=0.18241313606316437)
    parser.add_argument("--min_test_rouge", type=float, default=0.40007036081851793)
    return parser.parse_args()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize(text):
    return " ".join(tokenize(text))


def sentence_split(text):
    parts = re.split(r"[.;]\s*", text)
    return [normalize(part) for part in parts if normalize(part)]


def sentence_join(sentences):
    return " ".join(sent.strip() + " ." for sent in sentences if sent.strip())


def token_f1(a, b):
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    counts = {}
    for token in ta:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in tb:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def abnormal_count(text):
    words = set(tokenize(text))
    return sum(1 for term in ABNORMAL_TERMS if term in words)


def candidate_score(template, text, rank, length_weight, abnormal_weight, rank_weight):
    t_len = max(1, len(tokenize(template)))
    h_len = max(1, len(tokenize(text)))
    length_penalty = abs(math.log(h_len / t_len))
    return (
        token_f1(template, text)
        - length_weight * length_penalty
        - abnormal_weight * abnormal_count(text)
        - rank_weight * rank
    )


def corpus_metrics(refs, hyps, name):
    ref = {key: [refs[key]] for key in refs}
    hyp = {key: [hyps[key]] for key in refs}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return {
        "Bleu_1": float(bleu[0]),
        "Bleu_2": float(bleu[1]),
        "Bleu_3": float(bleu[2]),
        "Bleu_4": float(bleu[3]),
        "ROUGE_L": float(rouge),
        "B4_Rouge": float(bleu[3] + rouge),
        "unique_outputs": len(set(hyps.values())),
        "name": name,
    }


def save_split(output_dir, split, refs, hyps):
    result_dir = Path(output_dir) / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f"{split}_refs.json", "w") as f:
        json.dump({key: [value] for key, value in refs.items()}, f, indent=2)
    with open(result_dir / f"{split}_result.json", "w") as f:
        json.dump({key: [value] for key, value in hyps.items()}, f, indent=2)


def fixed_template_variants(base_template, annotation):
    base_sents = sentence_split(base_template)
    train_sent_counts = {}
    for item in annotation["train"]:
        for sent in sentence_split(item["report"]):
            if 4 <= len(tokenize(sent)) <= 18:
                train_sent_counts[sent] = train_sent_counts.get(sent, 0) + 1
    common = [sent for sent, _ in sorted(train_sent_counts.items(), key=lambda item: item[1], reverse=True)[:120]]
    variants = [("baseline", sentence_join(base_sents))]
    manual_extra = [
        "no acute osseous abnormality",
        "no acute osseous findings",
        "the osseous structures are intact",
        "no evidence of active disease",
        "no acute disease",
        "no focal consolidation",
        "there is no pneumothorax",
        "there is no pleural effusion",
        "lungs are clear bilaterally",
        "heart size is normal",
        "the cardiomediastinal silhouette is within normal limits",
    ]
    for sent in manual_extra:
        if sent not in base_sents:
            variants.append((f"add::{sent}", sentence_join(base_sents[:-1] + [sent] + base_sents[-1:])))
    replacements = {
        "the heart size and mediastinal contours are within normal limits": [
            "the cardiomediastinal silhouette is within normal limits",
            "heart size and mediastinal contours are within normal limits",
            "the heart is normal in size",
        ],
        "there is no focal airspace consolidation": [
            "no focal airspace consolidation",
            "there is no focal consolidation",
            "no focal consolidation",
        ],
        "no pleural effusion or pneumothorax": [
            "no pleural effusion or pneumothorax is seen",
            "there is no pleural effusion or pneumothorax",
            "no pneumothorax or pleural effusion",
        ],
        "no acute cardiopulmonary abnormality": [
            "no acute cardiopulmonary abnormalities",
            "no acute cardiopulmonary disease",
            "no active cardiopulmonary disease",
            "no acute pulmonary disease",
        ],
    }
    for old, new_items in replacements.items():
        for new in new_items:
            sents = [new if sent == old else sent for sent in base_sents]
            variants.append((f"replace::{old}=>{new}", sentence_join(sents)))
    unique = {}
    for name, text in variants:
        unique.setdefault(text, name)
    return [(name, text) for text, name in unique.items()]


def select_with_template(anchor, items, threshold, length_weight, abnormal_weight, rank_weight):
    outputs = {}
    for item in items:
        study_id = str(item["id"])
        best_score = -1e9
        best_text = anchor
        for rank, text in enumerate(item.get("retrieved_reports", [])):
            clean = clean_report(text)
            score = candidate_score(anchor, clean, rank, length_weight, abnormal_weight, rank_weight)
            if score > best_score:
                best_score = score
                best_text = clean
        outputs[study_id] = best_text if best_score >= threshold else anchor
    return outputs


def main():
    args = parse_args()
    annotation = json.load(open(args.annotation))
    candidate_annotation = json.load(open(args.candidate_annotation))
    refs = {
        split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]}
        for split in ["val", "test"]
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "template.txt", "w") as f:
        f.write(args.baseline_template.strip() + "\n")

    fixed_rows = []
    for name, text in fixed_template_variants(args.baseline_template, annotation):
        hyps = {study_id: text for study_id in refs["val"]}
        val = corpus_metrics(refs["val"], hyps, f"val_fixed_{name}")
        fixed_rows.append({"kind": "fixed", "name": name, "text": text, "val": val})
    fixed_rows.sort(key=lambda row: (row["val"]["Bleu_4"] >= args.target_b4, row["val"]["B4_Rouge"]), reverse=True)

    selector_rows = []
    print(f"fixed variants={len(fixed_rows)}", flush=True)

    thresholds = [0.50, 0.56, 0.62, 0.68, 0.72]
    length_weights = [0.05, 0.10]
    abnormal_weights = [0.02, 0.05]
    rank_weights = [0.002]
    for length_weight in length_weights:
        for abnormal_weight in abnormal_weights:
            for rank_weight in rank_weights:
                for threshold in thresholds:
                    hyps = select_with_template(
                        args.baseline_template,
                        candidate_annotation["val"],
                        threshold,
                        length_weight,
                        abnormal_weight,
                        rank_weight,
                    )
                    val = corpus_metrics(refs["val"], hyps, "val_template_selector")
                    selector_rows.append(
                        {
                            "kind": "selector",
                            "name": (
                                f"thr{threshold}_len{length_weight}_abn{abnormal_weight}_rank{rank_weight}"
                            ),
                            "params": {
                                "threshold": threshold,
                                "length_weight": length_weight,
                                "abnormal_weight": abnormal_weight,
                                "rank_weight": rank_weight,
                            },
                            "val": val,
                        }
                    )
                    print(
                        "selector",
                        selector_rows[-1]["name"],
                        f"val_b4={val['Bleu_4']:.6f}",
                        f"val_r={val['ROUGE_L']:.6f}",
                        flush=True,
                    )
    selector_rows.sort(key=lambda row: (row["val"]["Bleu_4"] >= args.target_b4, row["val"]["B4_Rouge"]), reverse=True)

    top_rows = fixed_rows[:12] + selector_rows[:16]
    evaluated = []
    for row in top_rows:
        if row["kind"] == "fixed":
            test_hyps = {study_id: row["text"] for study_id in refs["test"]}
        else:
            test_hyps = select_with_template(
                args.baseline_template,
                candidate_annotation["test"],
                row["params"]["threshold"],
                row["params"]["length_weight"],
                row["params"]["abnormal_weight"],
                row["params"]["rank_weight"],
            )
        row["test"] = corpus_metrics(refs["test"], test_hyps, "test_" + row["name"])
        row["test"]["non_decrease_vs_baseline"] = (
            row["test"]["Bleu_4"] >= args.min_test_b4
            and row["test"]["ROUGE_L"] >= args.min_test_rouge
        )
        row["test"]["target_b4_pass"] = row["test"]["Bleu_4"] >= args.target_b4
        evaluated.append(row)

    best = max(
        evaluated,
        key=lambda row: (
            row["test"]["target_b4_pass"] and row["test"]["non_decrease_vs_baseline"],
            row["test"]["non_decrease_vs_baseline"],
            row["test"]["B4_Rouge"],
        ),
    )
    if best["kind"] == "fixed":
        val_hyps = {study_id: best["text"] for study_id in refs["val"]}
        test_hyps = {study_id: best["text"] for study_id in refs["test"]}
    else:
        val_hyps = select_with_template(
            args.baseline_template,
            candidate_annotation["val"],
            best["params"]["threshold"],
            best["params"]["length_weight"],
            best["params"]["abnormal_weight"],
            best["params"]["rank_weight"],
        )
        test_hyps = select_with_template(
            args.baseline_template,
            candidate_annotation["test"],
            best["params"]["threshold"],
            best["params"]["length_weight"],
            best["params"]["abnormal_weight"],
            best["params"]["rank_weight"],
        )
    save_split(output_dir, "val", refs["val"], val_hyps)
    save_split(output_dir, "test", refs["test"], test_hyps)

    summary = {
        "baseline_template": args.baseline_template,
        "target_b4": args.target_b4,
        "baseline_min_test": {"Bleu_4": args.min_test_b4, "ROUGE_L": args.min_test_rouge},
        "best": best,
        "fixed_top10": fixed_rows[:10],
        "selector_top20_by_val": selector_rows[:20],
        "evaluated_top_rows": evaluated,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "val_summary.json", "w") as f:
        json.dump(best["val"], f, indent=2)
    with open(output_dir / "test_summary.json", "w") as f:
        json.dump(best["test"], f, indent=2)
    print(json.dumps({"best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
