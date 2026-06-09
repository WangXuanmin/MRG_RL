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


SMALL_GRID = {
    "beam_size": [3, 5],
    "length_penalty": [1.0, 1.5],
    "repetition_penalty": [1.0, 1.2],
    "no_repeat_ngram_size": [0, 2],
    "min_new_tokens": [10, 40],
    "max_new_tokens": [120],
}


def grid_items(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def run_one(delta_file, output_root, params, cuda_visible_devices):
    name = (
        f"beam{params['beam_size']}"
        f"_lp{params['length_penalty']}"
        f"_rp{params['repetition_penalty']}"
        f"_ng{params['no_repeat_ngram_size']}"
        f"_min{params['min_new_tokens']}"
        f"_max{params['max_new_tokens']}"
    ).replace(".", "p")
    save_dir = Path(output_root) / name
    result_file = save_dir / "result" / "test_result.json"
    log_file = save_dir / "run.log"
    summary_file = save_dir / "summary.json"

    if result_file.exists() and log_file.exists():
        print(f"[skip] {name}", flush=True)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = BASE_ARGS + [
            "--delta_file", delta_file,
            "--savedmodel_path", str(save_dir),
        ]
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        print("[run]", name, " ".join(cmd), flush=True)
        with open(log_file, "w") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
        if proc.returncode != 0:
            print(f"[fail] {name} exit={proc.returncode}", flush=True)
            return {"name": name, "status": "failed", "returncode": proc.returncode, **params}

    if not result_file.exists():
        return {"name": name, "status": "missing_result", **params}

    texts = load_hypotheses(result_file)
    summary = summarize_texts(texts)
    summary.update(parse_metrics(log_file))
    if "Bleu_4" in summary and "CIDEr" in summary:
        summary["combined"] = 0.5 * summary["Bleu_4"] + 0.5 * summary["CIDEr"]
    summary.update({"name": name, "status": "ok", **params})
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_file", required=True)
    parser.add_argument("--output_root", default="save/iu_xray/decode_sweep_sft_continue2")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--parallel_devices", default=None)
    parser.add_argument("--max_runs", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_jsonl = output_root / "summary.jsonl"
    params_list = []
    for idx, params in enumerate(grid_items(SMALL_GRID), start=1):
        if args.max_runs and idx > args.max_runs:
            break
        params_list.append(params)

    rows = []
    if args.parallel_devices:
        devices = [item.strip() for item in args.parallel_devices.split(",") if item.strip()]
        if not devices:
            raise ValueError("--parallel_devices did not contain any device ids")

        def run_indexed(item):
            index, params = item
            device = devices[index % len(devices)]
            return run_one(args.delta_file, output_root, params, device)

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(run_indexed, item) for item in enumerate(params_list)]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                with open(summary_jsonl, "a") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        for params in params_list:
            row = run_one(args.delta_file, output_root, params, args.cuda_visible_devices)
            rows.append(row)
            with open(summary_jsonl, "a") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: row.get("combined", -1), reverse=True)
    ranking = output_root / "ranking.md"
    with open(ranking, "w") as out:
        out.write("| rank | name | combined | Bleu_4 | CIDEr | ROUGE_L | unique | xxxx_mean |\n")
        out.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(ok_rows, start=1):
            out.write(
                f"| {rank} | {row['name']} | {row.get('combined', 0):.6f} | "
                f"{row.get('Bleu_4', 0):.6f} | {row.get('CIDEr', 0):.6f} | "
                f"{row.get('ROUGE_L', 0):.6f} | {row.get('unique_outputs', 0)} | "
                f"{row.get('xxxx_mean', 0):.6f} |\n"
            )
    print(f"wrote {ranking}", flush=True)
    if ok_rows:
        print("best", json.dumps(ok_rows[0], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
