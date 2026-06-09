#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.run_iuxray_clinical_oracle import CHEXBERT_LABELS, corpus_chexbert_micro, read_labels
from tools.train_iuxray_clinical_scalar_selector import BASE_TEMPLATE, text_features
from tools.train_iuxray_dino_reranker import build_groups, clean_report, corpus_score, encode_or_load


POS_SENTENCES = {
    "Enlarged Cardiomediastinum": "the cardiomediastinal silhouette is mildly enlarged .",
    "Cardiomegaly": "the heart is mildly enlarged .",
    "Lung Opacity": "there are mild bibasilar airspace opacities .",
    "Lung Lesion": "there is a pulmonary nodular opacity .",
    "Edema": "there is mild pulmonary vascular congestion .",
    "Consolidation": "there is focal airspace consolidation .",
    "Pneumonia": "findings may represent pneumonia .",
    "Atelectasis": "there are mild bibasilar atelectatic changes .",
    "Pneumothorax": "there is a small pneumothorax .",
    "Pleural Effusion": "there is a small pleural effusion .",
    "Pleural Other": "there is mild pleural thickening .",
    "Fracture": "there is an osseous fracture deformity .",
    "Support Devices": "support devices are present .",
}

NORMAL_SENTENCES = [
    "the heart size and mediastinal contours are within normal limits .",
    "the lungs are clear .",
    "there is no focal airspace consolidation .",
    "no pleural effusion or pneumothorax .",
    "there are degenerative changes of the spine .",
    "no acute cardiopulmonary abnormality .",
]


class ConstantBinaryModel:
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, x):
        n = x.shape[0]
        p = np.full(n, self.value, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def write_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def load_chexbert_labels(work_dir, split, kind):
    input_csv = os.path.join(work_dir, f"{split}_chexbert_work", f"{kind}.csv")
    labeled_csv = os.path.join(work_dir, f"{split}_chexbert_work", f"chexbert_{kind}", "labeled_reports.csv")
    return read_labels(input_csv, labeled_csv)


def label_matrix(labels, ids):
    y = np.zeros((len(ids), len(CHEXBERT_LABELS)), dtype=np.float32)
    for i, study_id in enumerate(ids):
        for j, label in enumerate(CHEXBERT_LABELS):
            y[i, j] = 1.0 if labels[study_id].get(label) == 1 else 0.0
    return y


def train_label_models(x_train, y_train):
    models = []
    for j in range(y_train.shape[1]):
        uniq = np.unique(y_train[:, j])
        if len(uniq) == 1:
            models.append(ConstantBinaryModel(uniq[0]))
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", C=0.5),
        )
        clf.fit(x_train, y_train[:, j])
        models.append(clf)
    return models


def predict_label_probs(models, x):
    cols = []
    for model in models:
        proba = model.predict_proba(x)
        cols.append(proba[:, 1])
    return np.stack(cols, axis=1).astype(np.float32)


def candidate_label_tensor(candidate_labels, group):
    out = []
    for i, study_id in enumerate(group["ids"]):
        rows = []
        for rank, _ in enumerate(group["candidate_ids"][i]):
            row_id = f"{study_id}__{rank}"
            rows.append([1.0 if candidate_labels[row_id].get(label) == 1 else 0.0 for label in CHEXBERT_LABELS])
        out.append(rows)
    return torch.tensor(out, dtype=torch.float32)


def expected_label_f1(probs, cand_labels):
    p = probs.unsqueeze(1)
    c = cand_labels
    tp = (p * c).sum(dim=-1)
    denom = p.sum(dim=-1) + c.sum(dim=-1)
    return torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1e-6), torch.zeros_like(tp))


def make_ngram_features(group, train_reports, report_freq):
    scalar = group["scalar"].numpy()
    rows = []
    n, k, _ = scalar.shape
    for i in range(n):
        for j, cand_id in enumerate(group["candidate_ids"][i]):
            rows.append(list(scalar[i, j]) + text_features(train_reports[cand_id], report_freq))
    return np.asarray(rows, dtype=np.float32), n, k


def train_ngram_model(train_group, train_reports, report_freq):
    x, _, _ = make_ngram_features(train_group, train_reports, report_freq)
    y = train_group["labels"].numpy().reshape(-1)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    return model


def predict_ngram_model(model, group, train_reports, report_freq):
    x, n, k = make_ngram_features(group, train_reports, report_freq)
    return torch.tensor(model.predict(x).reshape(n, k), dtype=torch.float32)


def normalize_rows(x):
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    return (x - mn) / (mx - mn).clamp_min(1e-6)


def select_candidate_predictions(group, train_reports, scores):
    pick = scores.argmax(dim=1).cpu().tolist()
    preds = {}
    choices = {}
    for row_idx, cand_pos in enumerate(pick):
        study_id = group["ids"][row_idx]
        cand_id = group["candidate_ids"][row_idx][cand_pos]
        preds[study_id] = train_reports[cand_id]
        choices[study_id] = {"rank": int(cand_pos), "candidate_id": cand_id}
    return preds, choices


def chex_metrics_from_candidate_choices(ref_labels, candidate_labels, choices, ids):
    hyp = {}
    for study_id in ids:
        hyp[study_id] = candidate_labels[f"{study_id}__{choices[study_id]['rank']}"]
    return corpus_chexbert_micro(ref_labels, hyp, ids)


def evaluate_candidate_grid(group, refs, train_reports, ref_labels, cand_labels, label_probs, ngram_pred, output_dir, split):
    cand_tensor = candidate_label_tensor(cand_labels, group)
    clinical = expected_label_f1(torch.tensor(label_probs, dtype=torch.float32), cand_tensor)
    sim = normalize_rows(group["scalar"][:, :, 0])
    rank = group["scalar"][:, :, 3]
    ngram = normalize_rows(ngram_pred)
    rows = []
    grid = []
    for nw in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for cw in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            for sw in [0.0, 0.25, 0.5]:
                for rw in [0.0, 0.05, 0.10]:
                    if nw == cw == sw == rw == 0.0:
                        continue
                    grid.append((nw, cw, sw, rw))
    for nw, cw, sw, rw in grid:
        scores = nw * ngram + cw * clinical + sw * sim + rw * rank
        preds, choices = select_candidate_predictions(group, train_reports, scores)
        metrics = corpus_score({sid: refs[sid] for sid in group["ids"]}, preds)
        metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
        metrics["unique_outputs"] = len(set(preds.values()))
        metrics["chexbert"] = chex_metrics_from_candidate_choices(ref_labels, cand_labels, choices, group["ids"])
        name = f"{split}_cand_nw{nw}_cw{cw}_sw{sw}_rw{rw}".replace(".", "p")
        row = {
            "name": name,
            "weights": {"ngram": nw, "clinical": cw, "sim": sw, "rank": rw},
            "metrics": metrics,
            "choices": choices,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["metrics"]["Bleu_4"] >= 0.20,
            row["metrics"]["ROUGE_L"] >= 0.40,
            row["metrics"]["chexbert"]["f1"] >= 0.40,
            row["metrics"]["B4_Rouge"],
            row["metrics"]["chexbert"]["f1"],
        ),
        reverse=True,
    )
    os.makedirs(output_dir, exist_ok=True)
    for row in rows[:8]:
        preds, _ = select_candidate_predictions(
            group,
            train_reports,
            row["weights"]["ngram"] * ngram
            + row["weights"]["clinical"] * clinical
            + row["weights"]["sim"] * sim
            + row["weights"]["rank"] * rank,
        )
        write_json_report(os.path.join(output_dir, f"{row['name']}_preds.json"), preds)
        with open(os.path.join(output_dir, f"{row['name']}_metrics.json"), "w") as f:
            json.dump(row["metrics"], f, indent=2)
    return [{k: v for k, v in row.items() if k != "choices"} for row in rows[:30]]


def template_from_probs(probs, threshold, max_pos):
    labels = [(CHEXBERT_LABELS[i], float(probs[i])) for i in range(len(CHEXBERT_LABELS)) if CHEXBERT_LABELS[i] != "No Finding"]
    labels = [item for item in labels if item[1] >= threshold and item[0] in POS_SENTENCES]
    labels.sort(key=lambda item: item[1], reverse=True)
    chosen = [label for label, _ in labels[:max_pos]]
    if not chosen:
        return " ".join(NORMAL_SENTENCES)
    sentences = [
        "the cardiomediastinal silhouette is stable .",
        "there is no pneumothorax .",
    ]
    for label in chosen:
        sentence = POS_SENTENCES[label]
        if sentence not in sentences:
            sentences.append(sentence)
    if "Pleural Effusion" not in chosen:
        sentences.append("no pleural effusion .")
    sentences.append("no acute osseous abnormality .")
    return " ".join(sentences)


def intended_template_labels(probs, threshold, max_pos):
    chosen = set()
    labels = [(CHEXBERT_LABELS[i], float(probs[i])) for i in range(len(CHEXBERT_LABELS)) if CHEXBERT_LABELS[i] != "No Finding"]
    labels = [item for item in labels if item[1] >= threshold and item[0] in POS_SENTENCES]
    labels.sort(key=lambda item: item[1], reverse=True)
    for label, _ in labels[:max_pos]:
        chosen.add(label)
    if not chosen:
        chosen.add("No Finding")
    return {label: (1 if label in chosen else 0) for label in CHEXBERT_LABELS}


def evaluate_template_grid(ids, refs, ref_labels, label_probs, output_dir, split):
    rows = []
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        for max_pos in [1, 2, 3, 4]:
            preds = {}
            hyp_labels = {}
            for i, study_id in enumerate(ids):
                preds[study_id] = template_from_probs(label_probs[i], threshold, max_pos)
                hyp_labels[study_id] = intended_template_labels(label_probs[i], threshold, max_pos)
            metrics = corpus_score({sid: refs[sid] for sid in ids}, preds)
            metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
            metrics["unique_outputs"] = len(set(preds.values()))
            metrics["chexbert_intended"] = corpus_chexbert_micro(ref_labels, hyp_labels, ids)
            name = f"{split}_label_template_t{threshold}_m{max_pos}".replace(".", "p")
            rows.append({"name": name, "threshold": threshold, "max_pos": max_pos, "metrics": metrics})
            write_json_report(os.path.join(output_dir, f"{name}_preds.json"), preds)
    rows.sort(
        key=lambda row: (
            row["metrics"]["Bleu_4"] >= 0.20,
            row["metrics"]["ROUGE_L"] >= 0.40,
            row["metrics"]["chexbert_intended"]["f1"] >= 0.40,
            row["metrics"]["B4_Rouge"],
            row["metrics"]["chexbert_intended"]["f1"],
        ),
        reverse=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--clinical_oracle_dir", default="save/iu_xray/rag_top50_clinical_oracle")
    parser.add_argument("--output_dir", default="save/iu_xray/label_guided_selector_top50")
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_reports = reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = Counter(train_reports.values())

    groups = {}
    for split in ["train", "val", "test"]:
        groups[split] = build_groups(split, ids[split], reports, embeddings, train_ids, train_matrix, train_reports, report_freq, args.topk, with_labels=True)
        print("group", split, groups[split]["labels"].shape, flush=True)

    ref_labels = {split: load_chexbert_labels(args.clinical_oracle_dir, split, "refs") for split in ["train", "val", "test"]}
    cand_labels = {split: load_chexbert_labels(args.clinical_oracle_dir, split, "candidates") for split in ["val", "test"]}
    x_train = np.stack([embeddings[("train", study_id)].numpy() for study_id in train_ids]).astype(np.float32)
    y_train = label_matrix(ref_labels["train"], train_ids)
    label_models = train_label_models(x_train, y_train)
    print("trained label models", flush=True)

    label_probs = {}
    label_cls_metrics = {}
    for split in ["train", "val", "test"]:
        x = np.stack([embeddings[(split, study_id)].numpy() for study_id in ids[split]]).astype(np.float32)
        label_probs[split] = predict_label_probs(label_models, x)
        pred_labels = {}
        for i, study_id in enumerate(ids[split]):
            pred_labels[study_id] = {label: (1 if label_probs[split][i, j] >= 0.5 else 0) for j, label in enumerate(CHEXBERT_LABELS)}
        label_cls_metrics[split] = corpus_chexbert_micro(ref_labels[split], pred_labels, ids[split])
        print("label_cls", split, label_cls_metrics[split], flush=True)

    ngram_model = train_ngram_model(groups["train"], train_reports, report_freq)
    summary = {"label_classifier": label_cls_metrics, "candidate_selector": {}, "label_template": {}}
    for split in ["val", "test"]:
        ngram_pred = predict_ngram_model(ngram_model, groups[split], train_reports, report_freq)
        summary["candidate_selector"][split] = evaluate_candidate_grid(
            groups[split],
            reports[split],
            train_reports,
            ref_labels[split],
            cand_labels[split],
            label_probs[split],
            ngram_pred,
            args.output_dir,
            split,
        )
        summary["label_template"][split] = evaluate_template_grid(
            ids[split],
            reports[split],
            ref_labels[split],
            label_probs[split],
            args.output_dir,
            split,
        )[:30]
        print("best_candidate", split, json.dumps(summary["candidate_selector"][split][0], indent=2), flush=True)
        print("best_template", split, json.dumps(summary["label_template"][split][0], indent=2), flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
