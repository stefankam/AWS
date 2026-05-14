"""Strictly modular data loading helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd
from datasets import load_dataset


def load_financial_dataset(dataset_name: str = "FinGPT/fingpt-sentiment-train") -> pd.DataFrame:
    ds = load_dataset(dataset_name)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    print(f"[data_loader] Loaded {len(df)} rows from {dataset_name} ({split}).")
    return df


def load_fingpt_market_data_via_api(symbol="AAPL", interval="1d", limit=30, base_url="http://localhost:8000", timeout=20):
    url = f"{base_url.rstrip('/')}/market-data"
    query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return pd.DataFrame(payload.get("data", payload))


def _first_existing(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def normalize_finance_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _first_existing(df, ["text", "sentence", "headline", "content", "title", "query", "input", "instruction", "output"])
    if text_col is None:
        raise ValueError(f"No text-like column found in {df.columns.tolist()}")

    label_col = _first_existing(df, ["label", "sentiment", "target", "output"])

    out = pd.DataFrame()
    if "instruction" in df.columns and "input" in df.columns:
        out["text"] = "Instruction: " + df["instruction"].astype(str) + "\nInput: " + df["input"].astype(str)
    else:
        out["text"] = df[text_col].astype(str)

    out["label"] = df[label_col].astype(str) if label_col else "unknown"

    if "date" in df.columns:
        out["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "timestamp" in df.columns:
        out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.date_range(end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0), periods=len(out), freq="h", tz="UTC")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out
