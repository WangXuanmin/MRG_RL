#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tools.run_iuxray_clinical_oracle import CHEXBERT_LABELS, corpus_chexbert_micro, read_labels
from tools.train_iuxray_dino_reranker import build_groups, clean_report, corpus_score, encode_or_load, select_predictions


class TextCrossScorer(nn.Module):
    def __init__(self, image_dim, text_dim, scalar_dim, hidden=512, proj=256, dropout=0.15):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(image_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(proj * 4 + scalar_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, image, text, scalar):
        qi = self.image_proj(image)
        ct = self.text_proj(text)
        x = torch.cat([qi, ct, qi * ct, torch.abs(qi - ct), scalar], dim=-1)
        return self.net(x).squeeze(-1)


def encode_reports_or_load(train_ids, train_reports, model_path, cache_path, batch_size):
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    output = {}
    with torch.no_grad():
        for start in range(0, len(train_ids), batch_size):
            ids = train_ids[start : start + batch_size]
            texts = [train_reports[study_id] for study_id in ids]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=192).to(device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            emb = F.normalize(emb, dim=-1).cpu()
            for study_id, vec in zip(ids, emb):
                output[study_id] = vec
            if (start // batch_size + 1) % 20 == 0:
                print(f"encoded reports {start + len(ids)}/{len(train_ids)}", flush=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(output, cache_path)
    return output


def load_candidate_scores(path):
    return json.load(open(path))


def make_label_tensor(group, candidate_scores, chex_weight):
    rows = []
    for study_id in group["ids"]:
        score_rows = candidate_scores[study_id]
        rows.append([row["B4_Rouge"] + chex_weight * row["chexbert_f1"] for row in score_rows])
    return torch.tensor(rows, dtype=torch.float32)


def group_text_tensor(group, report_embeddings):
    rows = []
    for cand_ids in group["candidate_ids"]:
        rows.append(torch.stack([report_embeddings[cand_id] for cand_id in cand_ids]))
    return torch.stack(rows)


def group_scores(model, group, text_tensor, device, batch_size):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, group["q"].shape[0], batch_size):
            q = group["q"][start : start + batch_size].to(device)
            t = text_tensor[start : start + batch_size].to(device)
            scalar = group["scalar"][start : start + batch_size].to(device)
            q_expand = q.unsqueeze(1).expand(-1, t.shape[1], -1)
            logits = model(
                q_expand.reshape(-1, q.shape[-1]),
                t.reshape(-1, t.shape[-1]),
                scalar.reshape(-1, scalar.shape[-1]),
            ).reshape(q.shape[0], t.shape[1])
            outs.append(logits.cpu())
    return torch.cat(outs, dim=0)


def write_json_report(path, reports):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({key: [value] for key, value in reports.items()}, f, indent=2)


def load_labels(work_dir, split, kind):
    return read_labels(
        os.path.join(work_dir, f"{split}_chexbert_work", f"{kind}.csv"),
        os.path.join(work_dir, f"{split}_chexbert_work", f"chexbert_{kind}", "labeled_reports.csv"),
    )


def candidate_label_metrics(ref_labels, cand_labels, group, scores):
    pick = scores.argmax(dim=1).cpu().tolist()
    hyp = {}
    for row_idx, cand_pos in enumerate(pick):
        study_id = group["ids"][row_idx]
        hyp[study_id] = cand_labels[f"{study_id}__{cand_pos}"]
    return corpus_chexbert_micro(ref_labels, hyp, group["ids"])


def evaluate(name, group, refs, train_reports, scores, output_dir, ref_labels=None, cand_labels=None):
    preds = select_predictions(group, scores, train_reports)
    metrics = corpus_score({sid: refs[sid] for sid in group["ids"]}, preds)
    metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
    metrics["unique_outputs"] = len(set(preds.values()))
    metrics["name"] = name
    if ref_labels is not None and cand_labels is not None:
        metrics["chexbert"] = candidate_label_metrics(ref_labels, cand_labels, group, scores)
    write_json_report(os.path.join(output_dir, f"{name}_preds.json"), preds)
    with open(os.path.join(output_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--vision_cache", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--text_model", default="resources/models/bert-base-uncased")
    parser.add_argument("--text_cache", default="save/iu_xray/text_cross_reranker/report_embeddings_bert.pt")
    parser.add_argument("--clinical_oracle_dir", default="save/iu_xray/rag_top50_clinical_oracle")
    parser.add_argument("--output_dir", default="save/iu_xray/text_cross_reranker_top50")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chex_weight", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.vision_cache, 64)
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

    report_embeddings = encode_reports_or_load(train_ids, train_reports, args.text_model, args.text_cache, 64)
    text_tensors = {split: group_text_tensor(groups[split], report_embeddings) for split in ["train", "val", "test"]}
    clinical_scores = {
        split: load_candidate_scores(os.path.join(args.clinical_oracle_dir, f"{split}_candidate_scores.json"))
        for split in ["train", "val", "test"]
    }
    train_targets = make_label_tensor(groups["train"], clinical_scores["train"], args.chex_weight)

    ref_labels = {split: load_labels(args.clinical_oracle_dir, split, "refs") for split in ["val", "test"]}
    cand_labels = {split: load_labels(args.clinical_oracle_dir, split, "candidates") for split in ["val", "test"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextCrossScorer(groups["train"]["q"].shape[-1], text_tensors["train"].shape[-1], groups["train"]["scalar"].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_train = groups["train"]["q"].shape[0]
    best = {"score": -1.0, "epoch": 0, "state": None, "metrics": None}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        accs = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start : start + args.batch_size]
            q = groups["train"]["q"][idx].to(device)
            t = text_tensors["train"][idx].to(device)
            scalar = groups["train"]["scalar"][idx].to(device)
            target_scores = train_targets[idx].to(device)
            q_expand = q.unsqueeze(1).expand(-1, t.shape[1], -1)
            logits = model(
                q_expand.reshape(-1, q.shape[-1]),
                t.reshape(-1, t.shape[-1]),
                scalar.reshape(-1, scalar.shape[-1]),
            ).reshape(q.shape[0], t.shape[1])
            soft = F.softmax(target_scores / args.tau, dim=1)
            loss = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(float((logits.argmax(dim=1) == target_scores.argmax(dim=1)).float().mean().detach().cpu()))

        val_scores = group_scores(model, groups["val"], text_tensors["val"], device, args.batch_size)
        val_metrics = evaluate(
            f"val_epoch{epoch}",
            groups["val"],
            reports["val"],
            train_reports,
            val_scores,
            args.output_dir,
            ref_labels["val"],
            cand_labels["val"],
        )
        epoch_score = val_metrics["Bleu_4"] + val_metrics["ROUGE_L"]
        item = {"epoch": epoch, "loss": float(np.mean(losses)), "train_hit": float(np.mean(accs)), "val": val_metrics}
        history.append(item)
        print(json.dumps(item, indent=2), flush=True)
        if epoch_score > best["score"]:
            best = {
                "score": epoch_score,
                "epoch": epoch,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": val_metrics,
            }
            torch.save(
                {"model": best["state"], "args": vars(args), "epoch": epoch, "metrics": val_metrics},
                os.path.join(args.output_dir, "best_text_cross_reranker.pth"),
            )

    model.load_state_dict(best["state"])
    final = {"history": history, "best_val": {"epoch": best["epoch"], "metrics": best["metrics"]}, "test": {}}
    for split in ["val", "test"]:
        scores = group_scores(model, groups[split], text_tensors[split], device, args.batch_size)
        final[split] = evaluate(
            f"{split}_best_epoch{best['epoch']}",
            groups[split],
            reports[split],
            train_reports,
            scores,
            args.output_dir,
            ref_labels.get(split),
            cand_labels.get(split),
        )
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
