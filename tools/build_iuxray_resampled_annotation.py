#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


ABNORMAL_TERMS = {
    "opacity",
    "effusion",
    "atelectasis",
    "cardiomegaly",
    "edema",
    "consolidation",
    "pneumothorax",
    "infiltrate",
    "nodule",
}


def clean_report(report):
    text = report.lower()
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_abnormal(report):
    text = clean_report(report)
    return any(term in text for term in ABNORMAL_TERMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--output", default="data/iu_xray/annotation_resampled_abnormal_cap.json")
    parser.add_argument("--template_cap", type=int, default=30)
    parser.add_argument("--abnormal_repeat", type=int, default=2)
    args = parser.parse_args()

    data = json.load(open(args.annotation))
    out = {key: list(value) for key, value in data.items()}
    counts = Counter()
    train = []
    for item in data["train"]:
        key = clean_report(item.get("report", ""))
        if counts[key] >= args.template_cap:
            continue
        counts[key] += 1
        repeat = args.abnormal_repeat if has_abnormal(item.get("report", "")) else 1
        if "xxxx" in key and not has_abnormal(item.get("report", "")):
            repeat = 1
        train.extend([item] * repeat)

    out["train"] = train
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "source_train": len(data["train"]),
        "output_train": len(train),
        "unique_clean_reports": len(counts),
        "template_cap": args.template_cap,
        "abnormal_repeat": args.abnormal_repeat,
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
