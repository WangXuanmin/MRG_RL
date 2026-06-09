#!/usr/bin/env python3
import json
import math
import re


BASE = (
    "the heart size and mediastinal contours are within normal limits . "
    "the lungs are clear . "
    "there is no focal airspace consolidation . "
    "no pleural effusion or pneumothorax . "
    "there are degenerative changes of the spine . "
    "no acute cardiopulmonary abnormality ."
)
ABNORMAL = {
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


def toks(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(a, b):
    ta = toks(a)
    tb = toks(b)
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


def score(text, rank, length_weight=0.05, abnormal_weight=0.02, rank_weight=0.002):
    length_penalty = abs(math.log(max(1, len(toks(text))) / max(1, len(toks(BASE)))))
    abnormal_penalty = sum(1 for term in ABNORMAL if term in set(toks(text)))
    return token_f1(BASE, text) - length_weight * length_penalty - abnormal_weight * abnormal_penalty - rank_weight * rank


def main():
    annotation = json.load(open("save/iu_xray/rag_top5_b4rouge/annotation_top5.json"))
    for split in ["val", "test"]:
        scores = []
        for item in annotation[split]:
            vals = [score(text, rank) for rank, text in enumerate(item.get("retrieved_reports", []))]
            scores.append(max(vals) if vals else -999)
        scores.sort()
        print(split)
        for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]:
            print(q, scores[min(len(scores) - 1, int(q * (len(scores) - 1)))])


if __name__ == "__main__":
    main()
