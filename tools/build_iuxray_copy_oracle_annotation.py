#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_iuxray_clinical_oracle import CHEXBERT_LABELS, corpus_chexbert_micro, read_labels
from tools.train_iuxray_dino_reranker import clean_report, corpus_score


def write_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def load_labels(work_dir, split, kind):
    return read_labels(
        os.path.join(work_dir, f"{split}_chexbert_work", f"{kind}.csv"),
        os.path.join(work_dir, f"{split}_chexbert_work", f"chexbert_{kind}", "labeled_reports.csv"),
    )


def select_rows(candidate_scores, ids, topk, chex_weight):
    out = {}
    for study_id in ids:
        rows = candidate_scores[study_id][:topk]
        best = max(rows, key=lambda row: row["B4_Rouge"] + chex_weight * row["chexbert_f1"])
        out[study_id] = best
    return out


def clinical_for_choices(ref_labels, cand_labels, choices, ids):
    hyp = {study_id: cand_labels[f"{study_id}__{choices[study_id]['rank']}"] for study_id in ids}
    return corpus_chexbert_micro(ref_labels, hyp, ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--candidate_annotation", default="save/iu_xray/rag_top50_b4rouge/annotation_top50.json")
    parser.add_argument("--clinical_oracle_dir", default="save/iu_xray/rag_top50_clinical_oracle")
    parser.add_argument("--output", default="data/iu_xray/annotation_rag_top20_copy_oracle_cw1.json")
    parser.add_argument("--summary", default="save/iu_xray/rag_top20_copy_oracle_cw1/build_summary.json")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--chex_weight", type=float, default=1.0)
    args = parser.parse_args()

    ann = json.load(open(args.annotation))
    cand_ann = json.load(open(args.candidate_annotation))
    output_ann = {}
    summary = {"topk": args.topk, "chex_weight": args.chex_weight, "splits": {}}
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)

    for split in ["train", "val", "test"]:
        ids = [str(item["id"]) for item in ann[split]]
        refs = {str(item["id"]): clean_report(item["report"]) for item in ann[split]}
        candidate_scores = json.load(open(os.path.join(args.clinical_oracle_dir, f"{split}_candidate_scores.json")))
        choices = select_rows(candidate_scores, ids, args.topk, args.chex_weight)
        pred_reports = {study_id: choices[study_id]["text"] for study_id in ids}
        metrics = corpus_score(refs, pred_reports)
        metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
        metrics["unique_outputs"] = len(set(pred_reports.values()))
        if split in ["val", "test"]:
            ref_labels = load_labels(args.clinical_oracle_dir, split, "refs")
            cand_labels = load_labels(args.clinical_oracle_dir, split, "candidates")
            metrics["chexbert"] = clinical_for_choices(ref_labels, cand_labels, choices, ids)
        summary["splits"][split] = {"oracle_within_topk": metrics}
        write_json_report(os.path.join(os.path.dirname(args.summary), f"{split}_oracle_preds.json"), pred_reports)

        output_ann[split] = []
        cand_by_id = {str(item["id"]): item for item in cand_ann[split]}
        for item in ann[split]:
            study_id = str(item["id"])
            new_item = dict(item)
            retrieved = cand_by_id[study_id].get("retrieved_reports", [])[: args.topk]
            new_item["retrieved_reports"] = retrieved
            if split == "train":
                new_item["report"] = choices[study_id]["text"]
                new_item["copy_oracle_rank"] = int(choices[study_id]["rank"])
                new_item["copy_oracle_score"] = float(choices[study_id]["B4_Rouge"] + args.chex_weight * choices[study_id]["chexbert_f1"])
            output_ann[split].append(new_item)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_ann, f, indent=2)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
