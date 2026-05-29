"""Evaluation helpers."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from metrics import compute_jains_fairness_index


def _causal_target_count(labels: torch.Tensor) -> int:
    """Count non-ignored next-token targets used by causal-LM loss."""
    if labels.shape[1] < 2:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def _avg_nll(
    model,
    tokenizer,
    texts: Iterable[str],
    max_seq_length: int = 128,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> float:
    rows = [str(t) for t in texts if str(t).strip()]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        return float("nan")

    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    dl = DataLoader(rows, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_texts in dl:
            enc = tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            token_count = _causal_target_count(labels)
            if token_count == 0:
                continue

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_nll += float(out.loss.item()) * token_count
            total_tokens += token_count

    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def evaluate_perplexity_by_persona(model, tokenizer, df, max_seq_length: int = 128, **kwargs):
    max_seq_length = int(kwargs.get("eval_max_seq_length", max_seq_length))
    batch_size = int(kwargs.get("batch_size", kwargs.get("eval_batch_size", 8)))
    max_samples = kwargs.get("max_samples", kwargs.get("eval_max_samples"))
    max_samples = int(max_samples) if max_samples is not None else None

    out = []
    for (persona, region), g in df.groupby(["persona", "region"]):
        nll = _avg_nll(
            model,
            tokenizer,
            g["text"].astype(str).tolist(),
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        ppl = float(math.exp(min(nll, 20.0))) if not np.isnan(nll) else float("nan")
        eval_count = min(len(g), max_samples) if max_samples is not None else len(g)
        out.append({"persona": persona, "region": region, "nll": nll, "perplexity": ppl, "num_eval_samples": eval_count})
    return pd.DataFrame(out, columns=["persona", "region", "nll", "perplexity", "num_eval_samples"])


def evaluate_term_perplexity(
    model,
    tokenizer,
    texts: Iterable[str],
    term: str,
    max_seq_length: int = 128,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> tuple[float, int]:
    """Evaluate perplexity on examples containing a target semantic-drift term."""
    term_l = str(term).lower()
    matched = [str(t) for t in texts if term_l in str(t).lower()]
    if max_samples is not None:
        matched = matched[:max_samples]
    nll = _avg_nll(
        model,
        tokenizer,
        matched,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        max_samples=None,
    )
    ppl = float(math.exp(min(nll, 20.0))) if not np.isnan(nll) else float("nan")
    return ppl, len(matched)


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
    labels[attention_mask == 0] = -100
    token_count = _causal_target_count(labels)
    if token_count == 0:
        return float("-inf")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return -float(out.loss.item()) * token_count


def evaluate_sentiment_accuracy_by_persona(model, tokenizer, df, max_seq_length: int = 128, **kwargs):
    max_seq_length = int(kwargs.get("eval_max_seq_length", max_seq_length))
    max_samples = kwargs.get("max_samples", kwargs.get("eval_max_samples"))
    max_samples = int(max_samples) if max_samples is not None else None
    if "label" not in df.columns or df.empty:
        return pd.DataFrame(columns=["persona", "accuracy"])

    work = df[["persona", "text", "label"]].copy()
    work["label"] = work["label"].astype(str).str.strip()
    work = work[work["label"] != ""]
    if max_samples is not None:
        work = work.head(max_samples)
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
