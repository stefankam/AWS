from __future__ import annotations
from pathlib import Path
import pandas as pd

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
        sub = sub.copy(); sub["label"] = sub[label_col].astype(str)
        sub[["text", "label", "timestamp", "persona", "region"]].to_csv(rd / f"{cid}.csv", index=False)
