#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/iu_xray/annotation_rag_top20_copy_oracle_cw1.json")
    parser.add_argument("--output", default="data/iu_xray/annotation_rag_top20_number_cw1.json")
    parser.add_argument("--summary", default="save/iu_xray/rag_top20_number_cw1/build_summary.json")
    args = parser.parse_args()

    ann = json.load(open(args.source))
    out = {}
    ranks = []
    for split, items in ann.items():
        out[split] = []
        for item in items:
            new_item = dict(item)
            if split == "train":
                rank = int(item["copy_oracle_rank"])
                ranks.append(rank)
                new_item["report"] = f"{rank + 1} ."
            out[split].append(new_item)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    summary = {
        "source": args.source,
        "output": args.output,
        "train_rank_histogram_zero_based": dict(sorted(Counter(ranks).items())),
        "train_size": len(ranks),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
