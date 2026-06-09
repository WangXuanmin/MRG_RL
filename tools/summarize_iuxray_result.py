#!/usr/bin/env python3
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


def load_hypotheses(path):
    data = json.load(open(path))
    texts = []
    for value in data.values():
        if isinstance(value, list) and value:
            texts.append(str(value[0]))
        elif isinstance(value, str):
            texts.append(value)
    return texts


def parse_metrics(log_path):
    if not log_path or not Path(log_path).exists():
        return {}
    text = Path(log_path).read_text(errors="replace")
    matches = re.findall(r"Test result of .*?: (\{.*?\})", text)
    if not matches:
        return {}
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return {}


def summarize_texts(texts):
    lengths = [len(text.split()) for text in texts]
    xxxx_rates = [
        text.split().count("xxxx") / max(1, len(text.split()))
        for text in texts
    ]
    counts = Counter(texts)
    return {
        "n": len(texts),
        "unique_outputs": len(counts),
        "len_mean": sum(lengths) / max(1, len(lengths)),
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "xxxx_mean": sum(xxxx_rates) / max(1, len(xxxx_rates)),
        "xxxx_max": max(xxxx_rates) if xxxx_rates else 0,
        "with_xxxx": sum("xxxx" in text for text in texts),
        "top_outputs": [
            {"count": count, "text": text}
            for text, count in counts.most_common(5)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    texts = load_hypotheses(args.result)
    summary = summarize_texts(texts)
    metrics = parse_metrics(args.log)
    summary.update(metrics)
    if "Bleu_4" in summary and "CIDEr" in summary:
        summary["combined"] = 0.5 * summary["Bleu_4"] + 0.5 * summary["CIDEr"]

    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n")


if __name__ == "__main__":
    main()
