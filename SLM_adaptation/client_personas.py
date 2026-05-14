"""Strictly modular persona assignment + synthetic client generation."""
from __future__ import annotations

import random
import numpy as np
import pandas as pd

CLIENT_PERSONAS = {
    "retail_investors": {"keywords": ["etf", "index fund", "retirement", "401k", "ira", "dividend"], "region": "US", "timezone_group": "US"},
    "crypto_traders": {"keywords": ["gm", "wagmi", "defi", "token", "memecoin", "hodl"], "region": "GLOBAL", "timezone_group": "GLOBAL"},
    "institutional_analysts": {"keywords": ["earnings", "guidance", "10-k", "10-q", "cpi", "yield curve"], "region": "US", "timezone_group": "US"},
    "european_users": {"keywords": ["ecb", "mifid", "esma", "dax", "cac 40", "euro stoxx"], "region": "EU", "timezone_group": "EU"},
    "asian_market_users": {"keywords": ["nikkei", "topix", "hang seng", "sse composite", "kospi"], "region": "ASIA", "timezone_group": "ASIA"},
    "macro_analysts": {"keywords": ["rates", "fomc", "inflation", "unemployment"], "region": "GLOBAL", "timezone_group": "GLOBAL"},
    "stock_pickers": {"keywords": ["valuation", "pe", "balance sheet", "guidance"], "region": "US", "timezone_group": "US"},
}


def assign_persona(text: str) -> str:
    tl = str(text).lower()
    scores = {p: sum(kw in tl for kw in cfg["keywords"]) for p, cfg in CLIENT_PERSONAS.items()}
    m = max(scores.values())
    return random.choice([p for p, s in scores.items() if s == m]) if m > 0 else random.choice(list(CLIENT_PERSONAS.keys()))


def create_clients(df: pd.DataFrame, num_clients: int = 100, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    work = df.copy()
    work["persona"] = work["text"].apply(assign_persona)

    clients = []
    personas = list(CLIENT_PERSONAS.keys())
    for cid in range(num_clients):
        persona = random.choice(personas)
        meta = CLIENT_PERSONAS[persona]
        clients.append({"client_id": f"client_{cid}", "persona": persona, "region": meta["region"], "timezone_group": meta["timezone_group"]})
    client_df = pd.DataFrame(clients)

    assigned = []
    for _, row in work.iterrows():
        matches = client_df[client_df["persona"] == row["persona"]]
        chosen = matches.sample(1, random_state=random.randint(0, 10**6)).iloc[0] if len(matches) else client_df.sample(1).iloc[0]
        assigned.append(chosen["client_id"])
    work["client_id"] = assigned
    work = work.merge(client_df, on="client_id", how="left", suffixes=("", "_client"))
    return work, client_df
