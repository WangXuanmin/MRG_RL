#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from tools.train_iuxray_dino_reranker import (
    build_groups,
    clean_report,
    corpus_score,
    encode_or_load,
    evaluate_selector,
)


def flatten_features(group, candidate_prior=None):
    scalar = group["scalar"].numpy()
    n, k, f = scalar.shape
    extra = np.zeros((n, k, 4), dtype=np.float32)
    if candidate_prior is not None:
        for i in range(n):
            for j, cand_id in enumerate(group["candidate_ids"][i]):
                prior = candidate_prior.get(cand_id, [0.0, 0.0, 0.0, 0.0])
                extra[i, j] = prior
    return np.concatenate([scalar, extra], axis=-1).reshape(n * k, f + 4)


def score_to_tensor(pred, n, k):
    return torch.tensor(pred.reshape(n, k), dtype=torch.float32)


def metric_payload(name, group, refs, train_reports, scores, output_dir):
    metrics = evaluate_selector(name, group, refs, train_reports, scores, output_dir)
    hit = float((scores.argmax(dim=1) == group["labels"].argmax(dim=1)).float().mean())
    metrics["candidate_hit"] = hit
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--output_dir", default="save/iu_xray/dino_scalar_reranker_b4rouge")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, 64)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_reports = reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = {}
    for text in train_reports.values():
        report_freq[text] = report_freq.get(text, 0) + 1

    groups = {}
    for split in ["train", "val", "test"]:
        groups[split] = build_groups(split, ids[split], reports, embeddings, train_ids, train_matrix, train_reports, report_freq, args.topk, with_labels=True)
        print("group", split, groups[split]["labels"].shape, flush=True)

    # Candidate-level prior learned from training groups only.
    values = {}
    for i, cand_ids in enumerate(groups["train"]["candidate_ids"]):
        for j, cand_id in enumerate(cand_ids):
            values.setdefault(cand_id, []).append(float(groups["train"]["labels"][i, j]))
    candidate_prior = {}
    for cand_id, vals in values.items():
        arr = np.asarray(vals, dtype=np.float32)
        candidate_prior[cand_id] = [
            float(arr.mean()),
            float(arr.max()),
            float(np.percentile(arr, 75)),
            float(np.log1p(len(arr)) / 5.0),
        ]

    x_train = flatten_features(groups["train"], candidate_prior)
    y_train = groups["train"]["labels"].numpy().reshape(-1)
    models = {
        "prior_only": None,
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.03, random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=0.75, n_jobs=-1, random_state=42),
    }

    final = {}
    for model_name, model in models.items():
        if model is not None:
            print("fit", model_name, flush=True)
            model.fit(x_train, y_train)
        final[model_name] = {}
        for split in ["val", "test"]:
            n, k = groups[split]["labels"].shape
            if model_name == "prior_only":
                pred = flatten_features(groups[split], candidate_prior)[:, -4]
            else:
                pred = model.predict(flatten_features(groups[split], candidate_prior))
            scores = score_to_tensor(pred, n, k)
            final[model_name][split] = metric_payload(f"{split}_{model_name}", groups[split], reports[split], train_reports, scores, args.output_dir)
            print(model_name, split, final[model_name][split], flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
