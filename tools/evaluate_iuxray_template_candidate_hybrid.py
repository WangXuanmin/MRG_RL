#!/usr/bin/env python3
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_iuxray_dino_reranker import clean_report, corpus_score

BASE = (
    "the heart size and mediastinal contours are within normal limits . "
    "the lungs are clear . "
    "there is no focal airspace consolidation . "
    "no pleural effusion or pneumothorax . "
    "there are degenerative changes of the spine . "
    "no acute cardiopulmonary abnormality ."
)


def write_json(path, reports):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: [v] for k, v in reports.items()}, f, indent=2)


def main():
    ann = json.load(open("data/iu_xray/annotation.json"))
    refs = {str(item["id"]): clean_report(item["report"]) for item in ann["test"]}
    ids = [str(item["id"]) for item in ann["test"]]
    output_dir = "save/iu_xray/template_candidate_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for pred_path in glob.glob("save/iu_xray/label_guided_selector_top50/test_cand_*_preds.json"):
        name = Path(pred_path).stem.replace("_preds", "")
        raw = json.load(open(pred_path))
        cand_preds = {sid: value[0] for sid, value in raw.items()}
        length_order = sorted(ids, key=lambda sid: len(cand_preds[sid].split()), reverse=True)
        for frac in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]:
            take = set(length_order[: int(len(ids) * frac)])
            hyps = {sid: (cand_preds[sid] if sid in take else BASE) for sid in ids}
            metrics = corpus_score(refs, hyps)
            metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
            metrics["unique_outputs"] = len(set(hyps.values()))
            metrics["name"] = name
            metrics["frac"] = frac
            rows.append(metrics)
            tag = f"{name}_longfrac{str(frac).replace('.', 'p')}"
            write_json(os.path.join(output_dir, f"{tag}_preds.json"), hyps)
            with open(os.path.join(output_dir, f"{tag}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
    rows.sort(
        key=lambda row: (
            row["Bleu_4"] >= 0.20,
            row["ROUGE_L"] >= 0.40,
            row["Bleu_4"] + row["ROUGE_L"],
        ),
        reverse=True,
    )
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    for row in rows[:30]:
        print(
            "{name} frac={frac} B4={Bleu_4:.6f} R={ROUGE_L:.6f} C={CIDEr:.6f} uniq={unique_outputs}".format(
                **row
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
