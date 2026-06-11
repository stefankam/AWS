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




def _safe_perplexity_from_nll(nll: float) -> float:
    return float(math.exp(min(nll, 20.0))) if not np.isnan(nll) else float("nan")


def _completion_nll(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    max_seq_length: int = 128,
) -> tuple[float, int]:
    """Score only completion tokens conditioned on a prompt."""
    prompt = str(prompt)
    completion = str(completion)
    if not completion.strip():
        return float("nan"), 0

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        return float("nan"), 0

    # Keep the scored completion in the context window and truncate only the
    # prompt prefix.  The drift snippet is appended at the end of each example,
    # so ordinary right-side truncation would often remove exactly the tokens we
    # need to score.
    max_completion = max(1, min(len(completion_ids), max_seq_length - 1))
    completion_ids = completion_ids[:max_completion]
    prompt_budget = max_seq_length - len(completion_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    ids = prompt_ids + completion_ids

    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    if input_ids.shape[1] < 2:
        return float("nan"), 0

    labels = input_ids.clone()
    prompt_len = len(prompt_ids)
    labels[:, :prompt_len] = -100
    token_count = _causal_target_count(labels)
    if token_count == 0:
        return float("nan"), 0

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item()), token_count


def _drift_prompt_completion(text: str, term: str) -> tuple[str, str] | None:
    """Split a drift example so only the new concept/snippet is scored."""
    text = str(text)
    term = str(term).strip()
    marker = "New concept:"
    lower = text.lower()
    marker_idx = lower.rfind(marker.lower())
    if marker_idx >= 0:
        return text[:marker_idx], text[marker_idx:]

    if term:
        term_idx = lower.find(term.lower())
        if term_idx >= 0:
            return text[:term_idx], text[term_idx:term_idx + len(term)]
    return None


def evaluate_drift_completion_perplexity(
    model,
    tokenizer,
    df,
    max_seq_length: int = 128,
    max_samples: int | None = None,
) -> tuple[float, float, int, int]:
    """Evaluate perplexity on only semantic-drift completion tokens.

    Full-text perplexity is dominated by ordinary finance-language tokens, so it
    can make a stale CrossLM prior look similar to federated models.  This metric
    masks the prompt/original note and scores only the appended drift concept or
    new-concept snippet, which is the part expected to improve from current local
    client updates.
    """
    if df.empty or "drift_concept" not in df.columns:
        return float("nan"), float("nan"), 0, 0

    work = df.copy()
    drift = work["drift_concept"].fillna("").astype(str).str.strip()
    work = work[drift != ""]
    if max_samples is not None:
        work = work.head(int(max_samples))
    if work.empty:
        return float("nan"), float("nan"), 0, 0

    total_nll = 0.0
    total_tokens = 0
    scored_examples = 0
    for _, row in work.iterrows():
        pair = _drift_prompt_completion(str(row["text"]), str(row.get("drift_concept", "")))
        if pair is None:
            continue
        nll, token_count = _completion_nll(
            model,
            tokenizer,
            pair[0],
            pair[1],
            max_seq_length=max_seq_length,
        )
        if token_count == 0 or np.isnan(nll):
            continue
        total_nll += nll * token_count
        total_tokens += token_count
        scored_examples += 1

    if total_tokens == 0:
        return float("nan"), float("nan"), 0, 0
    avg_nll = total_nll / total_tokens
    return avg_nll, _safe_perplexity_from_nll(avg_nll), scored_examples, total_tokens



def _drift_target_prompt_completion(text: str, term: str) -> tuple[str, str] | None:
    """Split a drift example so only the actual new term is scored."""
    text = str(text)
    term = str(term).strip()
    if not term:
        return None
    lower = text.lower()
    term_idx = lower.find(term.lower())
    if term_idx < 0:
        return None
    return text[:term_idx], text[term_idx:term_idx + len(term)]


def evaluate_drift_target_perplexity(
    model,
    tokenizer,
    df,
    max_seq_length: int = 128,
    max_samples: int | None = None,
) -> tuple[float, float, int, int]:
    """Evaluate NLL/perplexity on only the drift concept term.

    This is the stricter Option-B metric: instead of scoring the whole natural
    language drift sentence, it scores only the actual new/local concept token
    span recorded in ``drift_concept``.  That makes the metric less dominated by
    generic words like "risk", "liquidity", or "market" and more sensitive to
    whether the model has learned the client-only concept itself.
    """
    if df.empty or "drift_concept" not in df.columns:
        return float("nan"), float("nan"), 0, 0

    work = df.copy()
    drift = work["drift_concept"].fillna("").astype(str).str.strip()
    work = work[drift != ""]
    if max_samples is not None:
        work = work.head(int(max_samples))
    if work.empty:
        return float("nan"), float("nan"), 0, 0

    total_nll = 0.0
    total_tokens = 0
    scored_examples = 0
    for _, row in work.iterrows():
        pair = _drift_target_prompt_completion(str(row["text"]), str(row.get("drift_concept", "")))
        if pair is None:
            continue
        nll, token_count = _completion_nll(
            model,
            tokenizer,
            pair[0],
            pair[1],
            max_seq_length=max_seq_length,
        )
        if token_count == 0 or np.isnan(nll):
            continue
        total_nll += nll * token_count
        total_tokens += token_count
        scored_examples += 1

    if total_tokens == 0:
        return float("nan"), float("nan"), 0, 0
    avg_nll = total_nll / total_tokens
    return avg_nll, _safe_perplexity_from_nll(avg_nll), scored_examples, total_tokens


def _private_code_prompt(row) -> str:
    code = str(row.get("drift_private_code", row.get("drift_concept", ""))).strip()
    return f"New private market code: {code}. {code} signal label:"


def evaluate_private_code_choice(
    model,
    tokenizer,
    df,
    max_seq_length: int = 128,
    max_samples: int | None = None,
    choices: tuple[str, ...] = ("positive", "neutral", "negative"),
) -> dict:
    """Forced-choice evaluation for non-inferable private-code mappings.

    A row is scored only when it has both a private code and a target answer.
    The model receives a prompt containing the code and chooses the answer with
    lowest completion NLL.  This evaluates whether client-only code->label
    mappings were learned, instead of whether generic financial prose is fluent.
    """
    empty = {
        "private_code_accuracy": float("nan"),
        "private_code_margin": float("nan"),
        "private_code_correct_nll": float("nan"),
        "private_code_samples": 0,
    }
    required = {"drift_private_code", "drift_answer"}
    if df.empty or not required.issubset(df.columns):
        return empty

    work = df.copy()
    code = work["drift_private_code"].fillna("").astype(str).str.strip()
    answer = work["drift_answer"].fillna("").astype(str).str.strip().str.lower()
    work = work[(code != "") & (answer != "")]
    if max_samples is not None:
        work = work.head(int(max_samples))
    if work.empty:
        return empty

    correct = 0
    margins = []
    correct_nlls = []
    scored = 0
    valid_choices = tuple(str(c).strip().lower() for c in choices if str(c).strip())

    for _, row in work.iterrows():
        gold = str(row.get("drift_answer", "")).strip().lower()
        if gold not in valid_choices:
            continue
        prompt = _private_code_prompt(row)
        scores = {}
        for choice in valid_choices:
            nll, token_count = _completion_nll(
                model,
                tokenizer,
                prompt,
                f" {choice}",
                max_seq_length=max_seq_length,
            )
            if token_count > 0 and not np.isnan(nll):
                scores[choice] = nll
        if gold not in scores or not scores:
            continue
        pred = min(scores.items(), key=lambda kv: kv[1])[0]
        decoy_nlls = [v for k, v in scores.items() if k != gold]
        if not decoy_nlls:
            continue
        correct += int(pred == gold)
        correct_nlls.append(scores[gold])
        margins.append(min(decoy_nlls) - scores[gold])
        scored += 1

    if scored == 0:
        return empty
    return {
        "private_code_accuracy": float(correct / scored),
        "private_code_margin": float(np.mean(margins)) if margins else float("nan"),
        "private_code_correct_nll": (
            float(np.mean(correct_nlls)) if correct_nlls else float("nan")
        ),
        "private_code_samples": int(scored),
    }


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
        ppl = _safe_perplexity_from_nll(nll)
        eval_count = min(len(g), max_samples) if max_samples is not None else len(g)
        out.append(
            {
                "persona": persona,
                "region": region,
                "nll": nll,
                "perplexity": ppl,
                "num_eval_samples": eval_count,
            }
        )
    return pd.DataFrame(
        out, columns=["persona", "region", "nll", "perplexity", "num_eval_samples"]
    )

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
