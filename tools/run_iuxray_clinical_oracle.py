#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from tools.train_iuxray_dino_reranker import clean_report, corpus_score


CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Reference-side clinical-aware oracle over retrieved IU X-Ray candidates.")
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--candidate_annotation", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--chexbert_dir", default="resources/external/CheXbert/src")
    parser.add_argument("--chexbert_checkpoint", default="resources/models/chexbert/chexbert.pth")
    return parser.parse_args()


def pair_b4_rouge(ref_text, hyp_text):
    ref = {"x": [ref_text]}
    hyp = {"x": [hyp_text]}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return float(bleu[3]), float(rouge), float(bleu[3] + rouge)


def write_chexbert_input(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "Report Impression"])
        writer.writeheader()
        for row_id, report in rows:
            writer.writerow({"id": row_id, "Report Impression": report})


def run_chexbert(chexbert_dir, checkpoint, input_csv, output_dir):
    labeled_csv = os.path.join(output_dir, "labeled_reports.csv")
    if os.path.exists(labeled_csv):
        return labeled_csv
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "label.py",
        "--data",
        os.path.abspath(input_csv),
        "--output_dir",
        os.path.abspath(output_dir),
        "--checkpoint",
        os.path.abspath(checkpoint),
    ]
    subprocess.run(cmd, cwd=chexbert_dir, check=True)
    return labeled_csv


def read_labels(input_csv, labeled_csv):
    with open(input_csv, newline="") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    labels = {}
    with open(labeled_csv, newline="") as f:
        for row_id, row in zip(ids, csv.DictReader(f)):
            labels[row_id] = {}
            for label in CHEXBERT_LABELS:
                value = row.get(label, "")
                labels[row_id][label] = None if value in ["", "nan", "NaN"] else int(float(value))
    return labels


def f1_from_labels(ref_labels, hyp_labels):
    tp = fp = fn = 0
    for label in CHEXBERT_LABELS:
        ref_pos = ref_labels.get(label) == 1
        hyp_pos = hyp_labels.get(label) == 1
        if ref_pos and hyp_pos:
            tp += 1
        elif not ref_pos and hyp_pos:
            fp += 1
        elif ref_pos and not hyp_pos:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1


def corpus_chexbert_micro(ref_labels, hyp_labels, ids):
    tp = fp = fn = 0
    for study_id in ids:
        for label in CHEXBERT_LABELS:
            ref_pos = ref_labels[study_id].get(label) == 1
            hyp_pos = hyp_labels[study_id].get(label) == 1
            if ref_pos and hyp_pos:
                tp += 1
            elif not ref_pos and hyp_pos:
                fp += 1
            elif ref_pos and not hyp_pos:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def save_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ann = json.load(open(args.annotation))
    cand_ann = json.load(open(args.candidate_annotation))
    summary = {}

    for split in args.splits:
        refs = {str(item["id"]): clean_report(item["report"]) for item in ann[split]}
        cand_items = {str(item["id"]): item for item in cand_ann[split]}
        ids = [str(item["id"]) for item in ann[split]]

        ref_rows = [(study_id, refs[study_id]) for study_id in ids]
        cand_rows = []
        cand_text = {}
        for study_id in ids:
            for rank, report in enumerate(cand_items[study_id].get("retrieved_reports", [])):
                row_id = f"{study_id}__{rank}"
                text = clean_report(report)
                cand_rows.append((row_id, text))
                cand_text[row_id] = text

        work_dir = output_dir / f"{split}_chexbert_work"
        ref_csv = work_dir / "refs.csv"
        cand_csv = work_dir / "candidates.csv"
        write_chexbert_input(ref_csv, ref_rows)
        write_chexbert_input(cand_csv, cand_rows)
        ref_labeled = run_chexbert(
            args.chexbert_dir,
            args.chexbert_checkpoint,
            str(ref_csv),
            str(work_dir / "chexbert_refs"),
        )
        cand_labeled = run_chexbert(
            args.chexbert_dir,
            args.chexbert_checkpoint,
            str(cand_csv),
            str(work_dir / "chexbert_candidates"),
        )
        ref_labels = read_labels(ref_csv, ref_labeled)
        cand_labels = read_labels(cand_csv, cand_labeled)

        candidates_by_id = {}
        for study_id in ids:
            rows = []
            for rank, report in enumerate(cand_items[study_id].get("retrieved_reports", [])):
                row_id = f"{study_id}__{rank}"
                b4, rouge, ngram = pair_b4_rouge(refs[study_id], cand_text[row_id])
                chex = f1_from_labels(ref_labels[study_id], cand_labels[row_id])
                rows.append(
                    {
                        "rank": rank,
                        "text": cand_text[row_id],
                        "Bleu_4": b4,
                        "ROUGE_L": rouge,
                        "B4_Rouge": ngram,
                        "chexbert_f1": chex,
                    }
                )
            candidates_by_id[study_id] = rows

        split_rows = []
        for chex_weight in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
            for rouge_floor in [0.0, 0.30, 0.35, 0.40]:
                preds = {}
                pred_labels = {}
                choices = {}
                for study_id, rows in candidates_by_id.items():
                    viable = [row for row in rows if row["ROUGE_L"] >= rouge_floor]
                    if not viable:
                        viable = rows
                    best = max(
                        viable,
                        key=lambda row: row["B4_Rouge"] + chex_weight * row["chexbert_f1"],
                    )
                    preds[study_id] = best["text"]
                    pred_labels[study_id] = cand_labels[f"{study_id}__{best['rank']}"]
                    choices[study_id] = best
                metrics = corpus_score(refs, preds)
                metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
                metrics["unique_outputs"] = len(set(preds.values()))
                metrics["chexbert"] = corpus_chexbert_micro(ref_labels, pred_labels, ids)
                row = {
                    "name": f"cw{chex_weight}_rf{rouge_floor}",
                    "chex_weight": chex_weight,
                    "rouge_floor": rouge_floor,
                    "metrics": metrics,
                    "choices": choices,
                }
                split_rows.append(row)
                save_json_report(output_dir / f"{split}_{row['name']}_preds.json", preds)

        split_rows.sort(
            key=lambda row: (
                row["metrics"]["Bleu_4"] >= 0.20,
                row["metrics"]["ROUGE_L"] >= 0.40,
                row["metrics"]["chexbert"]["f1"] >= 0.40,
                row["metrics"]["chexbert"]["f1"],
                row["metrics"]["B4_Rouge"],
            ),
            reverse=True,
        )
        with open(output_dir / f"{split}_candidate_scores.json", "w") as f:
            json.dump(candidates_by_id, f, indent=2)
        with open(output_dir / f"{split}_summary.json", "w") as f:
            json.dump([{k: v for k, v in row.items() if k != "choices"} for row in split_rows], f, indent=2)
        summary[split] = [{k: v for k, v in row.items() if k != "choices"} for row in split_rows[:10]]
        print(split, json.dumps(summary[split][0], indent=2), flush=True)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
