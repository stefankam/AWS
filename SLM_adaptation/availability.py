from __future__ import annotations
import numpy as np
import pandas as pd

def availability_probability(timezone_group, round_id, num_rounds):
    hour = (round_id * 24 / max(1, num_rounds)) % 24
    if timezone_group == "ASIA":
        base = 0.75 if 1 <= hour <= 10 else 0.25
    elif timezone_group == "EU":
        base = 0.75 if 7 <= hour <= 17 else 0.25
    elif timezone_group == "US":
        base = 0.75 if 14 <= hour <= 23 else 0.25
    else:
        base = 0.65
    return base

def create_availability_matrix(client_metadata, num_rounds=50, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _, c in client_metadata.iterrows():
        chronic = rng.beta(5, 2)
        for r in range(num_rounds):
            p = availability_probability(c["timezone_group"], r, num_rounds)
            p = np.clip(p * chronic + rng.normal(0, 0.05), 0.05, 0.95)
            rows.append({"client_id": c["client_id"], "round_id": r, "round": r, "availability_probability": float(p), "availability_prob": float(p), "available": int(rng.random() < p)})
    return pd.DataFrame(rows)
