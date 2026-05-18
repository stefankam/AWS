"""Strictly modular data loading helpers backed by FinGPT text generation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def query_fingpt_with_prompt(
    prompt: str,
    model_name: str,
    max_new_tokens: int = 128,
) -> str:
    """Generate financial text via HF text-generation.

    model_name must be provided by config (no hardcoded distilgpt2 default).
    """

    from transformers import pipeline

    try:
        generator = pipeline("text-generation", model=model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to initialize generation model '{model_name}'. "
            "If this is gated/private, run `hf auth login` and ensure access is granted."
        ) from exc

    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]["generated_text"]



def load_financial_dataset(
    dataset_name: str = "fingpt_generate",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    adapter_model: str | None = None,
    samples_per_topic: int = 6,
) -> pd.DataFrame:
    """Build a dynamic df by prompting a generation model for finance text."""
    del dataset_name, adapter_model  # adapter tracked in config for experiment side
    topics = [
        "ETF approval and market reaction",
        "rate hikes and bond-equity rotation",
        "AI infrastructure earnings momentum",
        "MiCA regulation and EU crypto compliance",
        "central bank events and FX volatility",
        "small-cap emerging companies outlook",
    ]

    rows = []
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for ti, topic in enumerate(topics):
        for si in range(samples_per_topic):
            prompt = (
                "Write a concise financial assistant note (3-5 sentences) "
                f"about: {topic}. End with one token in brackets: [positive], [neutral], or [negative]."
            )
            text = query_fingpt_with_prompt(prompt, model_name=model_name, max_new_tokens=180)
            low = text.lower()
            label = "positive" if "[positive]" in low else "negative" if "[negative]" in low else "neutral"
            rows.append({
                "text": text,
                "label": label,
                "timestamp": now - pd.Timedelta(hours=(ti * samples_per_topic + si)),
                "topic": topic,
            })

    df = pd.DataFrame(rows)
    print(f"[data_loader] Generated {len(df)} rows using model '{model_name}'.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _first_existing(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def normalize_finance_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _first_existing(df, ["text", "sentence", "headline", "content", "title", "query", "input", "instruction", "output"])
    out = pd.DataFrame()
    if "instruction" in df.columns and "input" in df.columns:
        out["text"] = "Instruction: " + df["instruction"].astype(str) + "\nInput: " + df["input"].astype(str)
    elif text_col is not None:
        out["text"] = df[text_col].astype(str)
    else:
        out["text"] = df.astype(str).agg(" | ".join, axis=1)

    label_col = _first_existing(df, ["label", "sentiment", "target", "output"])
    out["label"] = df[label_col].astype(str) if label_col else "unknown"

    ts_col = _first_existing(df, ["date", "timestamp", "datetime", "published_at", "time"])
    if ts_col is not None:
        out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.date_range(
            end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
            periods=len(out),
            freq="h",
            tz="UTC",
        )

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out
