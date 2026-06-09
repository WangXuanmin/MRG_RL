#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tools.train_iuxray_dino_reranker import (
    build_groups,
    clean_report,
    corpus_score,
    encode_or_load,
    evaluate_selector,
)


BASE_TEMPLATE = (
    "the heart size and mediastinal contours are within normal limits . "
    "the lungs are clear . "
    "there is no focal airspace consolidation . "
    "no pleural effusion or pneumothorax . "
    "there are degenerative changes of the spine . "
    "no acute cardiopulmonary abnormality ."
)

ABNORMAL = [
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
    "pleural",
    "pneumothorax",
]
NORMAL_PHRASES = [
    "no acute cardiopulmonary",
    "no acute pulmonary",
    "no focal",
    "lungs are clear",
    "heart size",
    "within normal limits",
    "no pleural effusion",
    "no pneumothorax",
    "no active disease",
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(a, b):
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    counts = Counter(ta)
    overlap = 0
    for token in tb:
        if counts[token] > 0:
            overlap += 1
            counts[token] -= 1
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def text_features(text, report_freq):
    words = tokenize(text)
    word_set = set(words)
    lower = text.lower()
    length = len(words)
    return [
        length / 100.0,
        text.count(" .") / 10.0,
        words.count("xxxx") / max(1, length),
        np.log1p(report_freq.get(text, 0)) / 5.0,
        token_f1(BASE_TEMPLATE, text),
        sum(1 for term in ABNORMAL if term in word_set) / 10.0,
        sum(1 for phrase in NORMAL_PHRASES if phrase in lower) / 10.0,
        lower.count(" no ") / 10.0,
        lower.count("normal") / 10.0,
        lower.count("clear") / 10.0,
    ]


def load_candidate_scores(path):
    return json.load(open(path))


def make_xy(group, clinical_scores, train_reports, report_freq, label_chex_weight):
    scalar = group["scalar"].numpy()
    rows = []
    labels = []
    for i, study_id in enumerate(group["ids"]):
        score_rows = clinical_scores[study_id]
        for j, cand_id in enumerate(group["candidate_ids"][i]):
            text = train_reports[cand_id]
            feats = list(scalar[i, j]) + text_features(text, report_freq)
            rows.append(feats)
            row = score_rows[j]
            labels.append(row["B4_Rouge"] + label_chex_weight * row["chexbert_f1"])
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def predict_group(model, group, train_reports, report_freq):
    scalar = group["scalar"].numpy()
    rows = []
    n, k, _ = scalar.shape
    for i in range(n):
        for j, cand_id in enumerate(group["candidate_ids"][i]):
            text = train_reports[cand_id]
            rows.append(list(scalar[i, j]) + text_features(text, report_freq))
    pred = model.predict(np.asarray(rows, dtype=np.float32)).reshape(n, k)
    return torch.tensor(pred, dtype=torch.float32)


def metric_payload(name, group, refs, train_reports, scores, output_dir):
    metrics = evaluate_selector(name, group, refs, train_reports, scores, output_dir)
    metrics["candidate_hit_ngram"] = float((scores.argmax(dim=1) == group["labels"].argmax(dim=1)).float().mean())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--clinical_oracle_dir", default="save/iu_xray/rag_top50_clinical_oracle")
    parser.add_argument("--output_dir", default="save/iu_xray/clinical_scalar_selector_top50")
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
        groups[split] = build_groups(
            split,
            ids[split],
            reports,
            embeddings,
            train_ids,
            train_matrix,
            train_reports,
            report_freq,
            args.topk,
            with_labels=True,
        )
        print("group", split, groups[split]["labels"].shape, flush=True)

    clinical = {
        split: load_candidate_scores(os.path.join(args.clinical_oracle_dir, f"{split}_candidate_scores.json"))
        for split in ["train", "val", "test"]
    }
    final = {}
    for label_chex_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_train, y_train = make_xy(groups["train"], clinical["train"], train_reports, report_freq, label_chex_weight)
        models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "hgb": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05, random_state=42),
            "rf": RandomForestRegressor(n_estimators=400, min_samples_leaf=6, max_features=0.75, n_jobs=-1, random_state=42),
        }
        for model_name, model in models.items():
            key = f"{model_name}_cw{str(label_chex_weight).replace('.', 'p')}"
            print("fit", key, x_train.shape, flush=True)
            model.fit(x_train, y_train)
            final[key] = {}
            for split in ["val", "test"]:
                scores = predict_group(model, groups[split], train_reports, report_freq)
                final[key][split] = metric_payload(
                    f"{split}_{key}",
                    groups[split],
                    reports[split],
                    train_reports,
                    scores,
                    args.output_dir,
                )
                print(key, split, final[key][split], flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
