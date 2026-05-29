"""Evaluation helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from metrics import compute_jains_fairness_index




def create_available_training_stream(df_clients, availability_df):
    key = "round_id" if "round_id" in df_clients.columns else "round"
    merged = df_clients.merge(availability_df, on=["client_id", key], how="left")
    merged["available"] = merged["available"].fillna(0).astype(int)
    return merged[merged["available"] == 1].copy(), merged[merged["available"] == 0].copy()

def export_client_round_data(df, output_dir="financial_fl_data"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_col = next((c for c in ["label", "output", "label_x", "label_y"] if c in df.columns), None)
    work = df.copy()
    if label_col is None:
        work["label"] = "unknown"; label_col = "label"
    rcol = "round_id" if "round_id" in work.columns else "round"
    for (r, cid), sub in work.groupby([rcol, "client_id"]):
        rd = output_dir / f"round_{int(r)}"; rd.mkdir(parents=True, exist_ok=True)
        sub[["text", "label", "timestamp", "persona", "region"]].to_csv(rd / f"{cid}.csv", index=False)
        if "prompt" not in sub.columns:
            sub["prompt"] = ""
        sub[["prompt", "text", "label", "timestamp", "persona", "region"]].to_csv(rd / f"{cid}.csv", index=False)




def _avg_nll(model, tokenizer, texts: Iterable[str], max_seq_length: int = 128) -> float:
    rows = [str(t) for t in texts if str(t).strip()]
    if not rows:
        return float("nan")

    device = next(model.parameters()).device
    model.eval()
    losses = []

    with torch.no_grad():
        for text in rows:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                continue

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            losses.append(float(out.loss.item()))

    if not losses:
        return float("nan")
    return float(np.mean(losses))


def evaluate_perplexity_by_persona(model, tokenizer, df, max_seq_length: int = 128):
    out = []
    for (persona, region), g in df.groupby(["persona", "region"]):
        nll = _avg_nll(model, tokenizer, g["text"].astype(str).tolist(), max_seq_length=max_seq_length)
        ppl = float(math.exp(nll)) if not np.isnan(nll) else float("nan")
        out.append({"persona": persona, "region": region, "perplexity": ppl, "num_eval_samples": len(g)})
    return pd.DataFrame(out, columns=["persona", "region", "perplexity", "num_eval_samples"])


def _label_score(model, tokenizer, text: str, label: str, max_seq_length: int = 128) -> float:
    prompt = f"Text: {text}\nSentiment:"
    completion = f" {label}"
    full = prompt + completion

    full_enc = tokenizer(full, truncation=True, max_length=max_seq_length, return_tensors="pt")
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_seq_length, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)

    labels = input_ids.clone()
    prompt_len = int(prompt_enc["input_ids"].shape[1])
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return -float(out.loss.item())


def evaluate_sentiment_accuracy_by_persona(model, tokenizer, df, max_seq_length: int = 128):
    if "label" not in df.columns or df.empty:
        return pd.DataFrame(columns=["persona", "accuracy"])

    work = df[["persona", "text", "label"]].copy()
    work["label"] = work["label"].astype(str).str.strip()
    work = work[work["label"] != ""]
    if work.empty:
        return pd.DataFrame(columns=["persona", "accuracy"])

    label_space = sorted(work["label"].unique().tolist())
    if len(label_space) > 5:
        # Avoid excessively expensive label-scoring for noisy label spaces.
        label_space = work["label"].value_counts().head(5).index.tolist()
        work = work[work["label"].isin(label_space)]

    preds = []
    model.eval()
    for _, row in work.iterrows():
        scores = {
            lbl: _label_score(model, tokenizer, str(row["text"]), lbl, max_seq_length=max_seq_length)
            for lbl in label_space
        }
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        preds.append(pred)

    work = work.reset_index(drop=True)
    work["pred_label"] = preds
    work["correct"] = (work["pred_label"] == work["label"]).astype(float)

    acc = work.groupby("persona", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
    return acc


def summarize_fairness(per_persona_df):
    return {
        "worst_persona_perplexity": float(per_persona_df.perplexity.max()) if len(per_persona_df) else np.nan,
        "perplexity_variance": float(per_persona_df.perplexity.var()) if len(per_persona_df) else np.nan,
        "jain_fairness": compute_jains_fairness_index(1.0 / (per_persona_df.perplexity + 1e-8)) if len(per_persona_df) else np.nan,
    }
