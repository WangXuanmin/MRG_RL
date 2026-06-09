#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.rouge.rouge import Rouge


def clean_report(report):
    report_cleaner = lambda t: t.replace("..", ".").replace("..", ".").replace("..", ".").replace("1. ", "") \
        .replace(". 2. ", ". ").replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ") \
        .replace(" 2. ", ". ").replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ") \
        .strip().lower().split(". ")
    sent_cleaner = lambda t: re.sub(r"[.,?;*!%^&_+():\-\[\]{}]", "", t.replace("\"", "").replace("/", "")
                                .replace("\\", "").replace("'", "").strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    return " . ".join(tokens) + " ." if tokens else ""


def pair_score(ref_text, hyp_text):
    ref = {"x": [ref_text]}
    hyp = {"x": [hyp_text]}
    bleu, _ = Bleu(4).compute_score(ref, hyp)
    rouge, _ = Rouge().compute_score(ref, hyp)
    return float(bleu[3] + rouge), float(bleu[3]), float(rouge)


def corpus_score(refs, hypos):
    ref = {key: [refs[key]] for key in refs}
    hyp = {key: [hypos[key]] for key in refs}
    out = {}
    for scorer, method in [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]:
        score, _ = scorer.compute_score(ref, hyp)
        if isinstance(method, list):
            for name, value in zip(method, score):
                out[name] = float(value)
        else:
            out[method] = float(score)
    return out


def encode_or_load(annotation, base_dir, vision_model, cache_path, batch_size):
    if cache_path and os.path.exists(cache_path):
        print(f"loading embeddings from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    flat = []
    for split_name, items in annotation.items():
        for item in items:
            for image_path in item["image_path"]:
                flat.append((split_name, str(item["id"]), os.path.join(base_dir, image_path)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(vision_model)
    model = AutoModel.from_pretrained(vision_model).to(device).eval()
    sums = defaultdict(lambda: None)
    counts = defaultdict(int)

    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch = flat[start:start + batch_size]
            images = []
            for _, _, path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images, return_tensors="pt").to(device)
            embeds = model(**inputs).last_hidden_state.mean(dim=1)
            embeds = F.normalize(embeds.float(), dim=1).cpu()
            for (split_name, study_id, _), embed in zip(batch, embeds):
                key = (split_name, study_id)
                sums[key] = embed.clone() if sums[key] is None else sums[key] + embed
                counts[key] += 1
            if (start // batch_size + 1) % 20 == 0:
                print(f"encoded {start + len(batch)}/{len(flat)}", flush=True)

    embeddings = {key: F.normalize((value / counts[key]).unsqueeze(0), dim=1).squeeze(0) for key, value in sums.items()}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(embeddings, cache_path)
    return embeddings


def report_stats(text, freq):
    words = text.split()
    sent_count = max(1, text.count(" ."))
    return [
        len(words) / 100.0,
        sent_count / 10.0,
        words.count("xxxx") / max(1, len(words)),
        np.log1p(freq) / 5.0,
    ]


class PairScorer(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + 8, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, q, c, scalar):
        x = torch.cat([q, c, q * c, torch.abs(q - c), scalar], dim=-1)
        return self.net(x).squeeze(-1)


def build_groups(split, ids, reports, embeddings, train_ids, train_matrix, train_reports, report_freq, topk, with_labels):
    q_embeds = []
    cand_indices = []
    scalar_feats = []
    labels = []
    candidate_ids = []

    for study_id in ids:
        q = embeddings[(split, study_id)]
        sims = train_matrix @ q
        raw_order = sims.topk(min(topk + (1 if split == "train" else 0), len(train_ids))).indices.tolist()
        order = []
        for rank in raw_order:
            cand_id = train_ids[rank]
            if split == "train" and cand_id == study_id:
                continue
            order.append(rank)
            if len(order) >= topk:
                break
        q_embeds.append(q)
        cand_indices.append(order)
        candidate_ids.append([train_ids[i] for i in order])
        top_sim = float(sims[order[0]])
        sf = []
        row_labels = []
        for rank_pos, train_idx in enumerate(order):
            cand_id = train_ids[train_idx]
            cand_text = train_reports[cand_id]
            sim = float(sims[train_idx])
            base = [
                sim,
                top_sim - sim,
                rank_pos / max(1, topk - 1),
                1.0 / (rank_pos + 1),
            ]
            base.extend(report_stats(cand_text, report_freq[cand_text]))
            sf.append(base)
            if with_labels:
                reward, _, _ = pair_score(reports[split][study_id], cand_text)
                row_labels.append(reward)
        scalar_feats.append(sf)
        if with_labels:
            labels.append(row_labels)

    payload = {
        "q": torch.stack(q_embeds),
        "cand_indices": torch.tensor(cand_indices, dtype=torch.long),
        "scalar": torch.tensor(scalar_feats, dtype=torch.float32),
        "candidate_ids": candidate_ids,
        "ids": ids,
    }
    if with_labels:
        payload["labels"] = torch.tensor(labels, dtype=torch.float32)
    return payload


def select_predictions(group, scores, train_reports):
    pick = scores.argmax(dim=1).cpu().tolist()
    preds = {}
    for row_idx, cand_pos in enumerate(pick):
        study_id = group["ids"][row_idx]
        cand_id = group["candidate_ids"][row_idx][cand_pos]
        preds[study_id] = train_reports[cand_id]
    return preds


def evaluate_selector(name, group, refs, train_reports, scores, output_dir):
    preds = select_predictions(group, scores, train_reports)
    metrics = corpus_score({sid: refs[sid] for sid in group["ids"]}, preds)
    metrics["B4_Rouge"] = metrics["Bleu_4"] + metrics["ROUGE_L"]
    texts = list(preds.values())
    metrics["unique_outputs"] = len(set(texts))
    metrics["name"] = name
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_preds.json"), "w") as f:
        json.dump({k: [v] for k, v in preds.items()}, f, indent=2)
    with open(os.path.join(output_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def group_scores(model, group, train_matrix, device, batch_size):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, group["q"].shape[0], batch_size):
            q = group["q"][start:start + batch_size].to(device)
            idx = group["cand_indices"][start:start + batch_size]
            c = train_matrix[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], -1).to(device)
            q_expand = q.unsqueeze(1).expand_as(c)
            scalar = group["scalar"][start:start + batch_size].to(device)
            score = model(q_expand.reshape(-1, q.shape[-1]), c.reshape(-1, q.shape[-1]), scalar.reshape(-1, scalar.shape[-1]))
            outs.append(score.reshape(idx.shape[0], idx.shape[1]).cpu())
    return torch.cat(outs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/iu_xray/annotation.json")
    parser.add_argument("--base_dir", default="data/iu_xray/images")
    parser.add_argument("--vision_model", default="resources/models/facebook_dinov2-base")
    parser.add_argument("--output_dir", default="save/iu_xray/dino_reranker_b4rouge")
    parser.add_argument("--cache_path", default="save/iu_xray/dino_reranker_b4rouge/study_embeddings.pt")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    annotation = json.load(open(args.annotation))
    embeddings = encode_or_load(annotation, args.base_dir, args.vision_model, args.cache_path, args.encode_batch_size)
    reports = {split: {str(item["id"]): clean_report(item["report"]) for item in annotation[split]} for split in ["train", "val", "test"]}
    ids = {split: [str(item["id"]) for item in annotation[split]] for split in ["train", "val", "test"]}
    train_ids = ids["train"]
    train_reports = reports["train"]
    train_matrix = torch.stack([embeddings[("train", study_id)] for study_id in train_ids])
    report_freq = Counter(train_reports.values())

    print("building groups", flush=True)
    groups = {}
    for split in ["train", "val", "test"]:
        groups[split] = build_groups(split, ids[split], reports, embeddings, train_ids, train_matrix, train_reports, report_freq, args.topk, with_labels=True)
        print(split, groups[split]["q"].shape, groups[split]["cand_indices"].shape, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_matrix_device = train_matrix.to(device)
    model = PairScorer(train_matrix.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_train = groups["train"]["q"].shape[0]
    best = {"val_score": -1, "epoch": None, "state": None, "metrics": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        accs = []
        for start in range(0, n_train, args.train_batch_size):
            batch_idx = perm[start:start + args.train_batch_size]
            q = groups["train"]["q"][batch_idx].to(device)
            idx = groups["train"]["cand_indices"][batch_idx]
            c = train_matrix_device[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], -1)
            q_expand = q.unsqueeze(1).expand_as(c)
            scalar = groups["train"]["scalar"][batch_idx].to(device)
            labels = groups["train"]["labels"][batch_idx].to(device)
            logits = model(q_expand.reshape(-1, q.shape[-1]), c.reshape(-1, q.shape[-1]), scalar.reshape(-1, scalar.shape[-1])).reshape(idx.shape[0], idx.shape[1])
            target = labels.argmax(dim=1)
            soft = F.softmax(labels / 0.08, dim=1)
            ce = F.cross_entropy(logits, target)
            kl = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
            mse = F.mse_loss(torch.sigmoid(logits), labels / 2.0)
            loss = ce + 0.3 * kl + 0.05 * mse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(float((logits.argmax(dim=1) == target).float().mean().detach().cpu()))

        val_logits = group_scores(model, groups["val"], train_matrix_device, device, args.train_batch_size)
        val_hit = float((val_logits.argmax(dim=1) == groups["val"]["labels"].argmax(dim=1)).float().mean())
        val_metrics = evaluate_selector(f"val_epoch{epoch}", groups["val"], reports["val"], train_reports, val_logits, args.output_dir)
        epoch_log = {"epoch": epoch, "loss": float(np.mean(losses)), "train_hit": float(np.mean(accs)), "val_hit": val_hit, "val_metrics": val_metrics}
        print(json.dumps(epoch_log, indent=2), flush=True)
        if val_metrics["B4_Rouge"] > best["val_score"]:
            best = {"val_score": val_metrics["B4_Rouge"], "epoch": epoch, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "metrics": val_metrics}
            torch.save({"model": best["state"], "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, os.path.join(args.output_dir, "best_reranker.pt"))

    model.load_state_dict(best["state"])
    final = {"best_epoch": best["epoch"], "best_val": best["metrics"]}

    for split in ["val", "test"]:
        top1_scores = torch.zeros(groups[split]["q"].shape[0], args.topk)
        top1_scores[:, 0] = 1.0
        oracle_scores = groups[split]["labels"]
        learned_scores = group_scores(model, groups[split], train_matrix_device, device, args.train_batch_size)
        final[f"{split}_top1"] = evaluate_selector(f"{split}_top1", groups[split], reports[split], train_reports, top1_scores, args.output_dir)
        final[f"{split}_oracle_top{args.topk}"] = evaluate_selector(f"{split}_oracle_top{args.topk}", groups[split], reports[split], train_reports, oracle_scores, args.output_dir)
        final[f"{split}_reranker"] = evaluate_selector(f"{split}_reranker", groups[split], reports[split], train_reports, learned_scores, args.output_dir)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
