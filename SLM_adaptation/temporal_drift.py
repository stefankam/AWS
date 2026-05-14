from __future__ import annotations
import pandas as pd

def add_temporal_rounds(df: pd.DataFrame, num_rounds: int = 50) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["round_id"] = pd.qcut(out.index, q=num_rounds, labels=False, duplicates="drop").astype(int)
    out["round"] = out["round_id"]
    return out
