#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evalcap.llm_judge import load_report_json


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
    parser = argparse.ArgumentParser(description="Evaluate CheXbert label F1 and RadGraph F1 for IU X-Ray reports.")
    parser.add_argument("--refs", required=True)
    parser.add_argument("--hyps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--chexbert_dir", default="resources/external/CheXbert/src")
    parser.add_argument("--chexbert_checkpoint", default="resources/models/chexbert/chexbert.pth")
    parser.add_argument("--radgraph_cache_dir", default="resources/models/radgraph")
    parser.add_argument("--radgraph_tar", default="resources/models/radgraph/modern-radgraph-xl.tar.gz")
    parser.add_argument("--radgraph_model_type", default="modern-radgraph-xl")
    parser.add_argument("--radgraph_tokenizer_dir", default="resources/models/ModernBERT-base")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--skip_chexbert", action="store_true")
    parser.add_argument("--skip_radgraph", action="store_true")
    return parser.parse_args()


def write_chexbert_input(path, ids, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "Report Impression"])
        writer.writeheader()
        for study_id in ids:
            writer.writerow({"id": study_id, "Report Impression": reports[study_id]})


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


def read_chexbert_output(input_csv, labeled_csv):
    with open(input_csv, newline="") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    labels = {}
    with open(labeled_csv, newline="") as f:
        for study_id, row in zip(ids, csv.DictReader(f)):
            labels[study_id] = {}
            for label in CHEXBERT_LABELS:
                value = row.get(label, "")
                if value in ["", "nan", "NaN"]:
                    labels[study_id][label] = None
                else:
                    labels[study_id][label] = int(float(value))
    return labels


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def chexbert_scores(ref_labels, hyp_labels, ids):
    total_tp = total_fp = total_fn = 0
    per_label = {}
    for label in CHEXBERT_LABELS:
        tp = fp = fn = 0
        for study_id in ids:
            ref_pos = ref_labels[study_id].get(label) == 1
            hyp_pos = hyp_labels[study_id].get(label) == 1
            if ref_pos and hyp_pos:
                tp += 1
            elif not ref_pos and hyp_pos:
                fp += 1
            elif ref_pos and not hyp_pos:
                fn += 1
        per_label[label] = f1_from_counts(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    macro_f1 = sum(row["f1"] for row in per_label.values()) / len(per_label)
    return {
        "micro": f1_from_counts(total_tp, total_fp, total_fn),
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def ensure_radgraph_cache(cache_dir, tar_path, model_type):
    model_dir = os.path.join(cache_dir, model_type)
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return
    os.makedirs(model_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)


def compute_radgraph(refs, hyps, ids, cache_dir, tar_path, model_type, tokenizer_dir):
    ensure_radgraph_cache(cache_dir, tar_path, model_type)
    from radgraph import F1RadGraph

    scorer = F1RadGraph(
        reward_level="all",
        model_type=model_type,
        model_cache_dir=cache_dir,
        tokenizer_cache_dir=tokenizer_dir,
    )
    mean_reward, reward_list, _, _ = scorer(
        hyps=[hyps[study_id] for study_id in ids],
        refs=[refs[study_id] for study_id in ids],
    )
    names = ["rg_e", "rg_er", "rg_bar_er"]
    return {
        "mean": {name: float(value) for name, value in zip(names, mean_reward)},
        "per_item": {
            study_id: {name: float(reward_list[index][row]) for index, name in enumerate(names)}
            for row, study_id in enumerate(ids)
        },
    }


def main():
    args = parse_args()
    refs = load_report_json(args.refs)
    hyps = load_report_json(args.hyps)
    ids = sorted(set(refs) & set(hyps))
    if args.limit_samples:
        ids = ids[: args.limit_samples]
    refs = {study_id: refs[study_id] for study_id in ids}
    hyps = {study_id: hyps[study_id] for study_id in ids}

    os.makedirs(args.work_dir, exist_ok=True)
    result = {"num_samples": len(ids)}

    if not args.skip_chexbert:
        ref_csv = os.path.join(args.work_dir, "refs.csv")
        hyp_csv = os.path.join(args.work_dir, "hyps.csv")
        write_chexbert_input(ref_csv, ids, refs)
        write_chexbert_input(hyp_csv, ids, hyps)
        ref_labeled = run_chexbert(
            args.chexbert_dir,
            args.chexbert_checkpoint,
            ref_csv,
            os.path.join(args.work_dir, "chexbert_refs"),
        )
        hyp_labeled = run_chexbert(
            args.chexbert_dir,
            args.chexbert_checkpoint,
            hyp_csv,
            os.path.join(args.work_dir, "chexbert_hyps"),
        )
        result["chexbert"] = chexbert_scores(
            read_chexbert_output(ref_csv, ref_labeled),
            read_chexbert_output(hyp_csv, hyp_labeled),
            ids,
        )
        result["chexbert"]["ref_labels_csv"] = ref_labeled
        result["chexbert"]["hyp_labels_csv"] = hyp_labeled

    if not args.skip_radgraph:
        result["radgraph"] = compute_radgraph(
            refs,
            hyps,
            ids,
            args.radgraph_cache_dir,
            args.radgraph_tar,
            args.radgraph_model_type,
            args.radgraph_tokenizer_dir,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, indent=2))


if __name__ == "__main__":
    main()
