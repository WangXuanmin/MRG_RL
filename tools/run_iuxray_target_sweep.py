#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
import subprocess
from pathlib import Path

from summarize_iuxray_result import parse_metrics, summarize_texts, load_hypotheses


BASE_ARGS = [
    "python", "-u", "train.py",
    "--test",
    "--dataset", "iu_xray",
    "--annotation", "data/iu_xray/annotation.json",
    "--base_dir", "data/iu_xray/images",
    "--vision_model", "resources/models/facebook_dinov2-base",
    "--llm_model", "resources/models/Qwen3-8B",
    "--llm_use_lora", "True",
    "--llm_r", "16",
    "--llm_alpha", "32",
    "--resampler_num_queries", "32",
    "--resampler_num_layers", "2",
    "--test_batch_size", "16",
    "--max_length", "160",
    "--freeze_vm", "True",
    "--num_workers", "8",
    "--devices", "1",
    "--strategy", "auto",
    "--accelerator", "gpu",
    "--precision", "bf16-mixed",
]


TARGET_GRID = {
    "beam_size": [2, 3, 5, 8],
    "length_penalty": [0.8, 1.0, 1.5, 2.0],
    "repetition_penalty": [1.0, 1.5],
    "no_repeat_ngram_size": [0, 2],
    "min_new_tokens": [40],
    "max_new_tokens": [120, 160],
}


def iter_grid():
    keys = list(TARGET_GRID)
    for values in itertools.product(*(TARGET_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def name_for(params):
    return (
        f"beam{params['beam_size']}"
        f"_lp{params['length_penalty']}"
        f"_rp{params['repetition_penalty']}"
        f"_ng{params['no_repeat_ngram_size']}"
        f"_min{params['min_new_tokens']}"
        f"_max{params['max_new_tokens']}"
    ).replace(".", "p")


def run_one(delta_file, output_root, params, device):
    name = name_for(params)
    save_dir = Path(output_root) / name
    result_file = save_dir / "result" / "test_result.json"
    log_file = save_dir / "run.log"
    summary_file = save_dir / "summary.json"

    if not (result_file.exists() and log_file.exists()):
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = BASE_ARGS + ["--delta_file", delta_file, "--savedmodel_path", str(save_dir)]
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        print("[run]", device, name, flush=True)
        with open(log_file, "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
        if proc.returncode != 0:
            return {"name": name, "status": "failed", "returncode": proc.returncode, **params}
    else:
        print("[skip]", name, flush=True)

    if not result_file.exists():
        return {"name": name, "status": "missing_result", **params}

    texts = load_hypotheses(result_file)
    summary = summarize_texts(texts)
    summary.update(parse_metrics(log_file))
    summary.update({"name": name, "status": "ok", **params})
    summary["target_score"] = summary.get("Bleu_4", 0.0) + summary.get("ROUGE_L", 0.0)
    summary["target_pass"] = summary.get("Bleu_4", 0.0) >= 0.17 and summary.get("ROUGE_L", 0.0) >= 0.38
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def write_ranking(output_root):
    rows = []
    for file in Path(output_root).glob("*/summary.json"):
        rows.append(json.load(open(file)))
    rows.sort(key=lambda row: row.get("target_score", -1), reverse=True)
    with open(Path(output_root) / "ranking_target.md", "w") as out:
        out.write("| rank | name | B4 | Rouge-L | target_score | CIDEr | unique | xxxx_mean | pass |\n")
        out.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        for idx, row in enumerate(rows, start=1):
            out.write(
                f"| {idx} | {row['name']} | {row.get('Bleu_4', 0):.6f} | "
                f"{row.get('ROUGE_L', 0):.6f} | {row.get('target_score', 0):.6f} | "
                f"{row.get('CIDEr', 0):.6f} | {row.get('unique_outputs', 0)} | "
                f"{row.get('xxxx_mean', 0):.6f} | {row.get('target_pass', False)} |\n"
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--parallel_devices", default="0")
    parser.add_argument("--max_runs", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    params_list = list(iter_grid())
    if args.max_runs:
        params_list = params_list[: args.max_runs]
    devices = [item.strip() for item in args.parallel_devices.split(",") if item.strip()]

    rows = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = []
        for idx, params in enumerate(params_list):
            futures.append(pool.submit(run_one, args.delta_file, output_root, params, devices[idx % len(devices)]))
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with open(output_root / "summary_target.jsonl", "a") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            write_ranking(output_root)

    ranked = write_ranking(output_root)
    if ranked:
        print("best", json.dumps(ranked[0], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
