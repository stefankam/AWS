"""Client scheduling baselines."""
from __future__ import annotations
import numpy as np

def random_scheduler(available_clients, k, rng):
    if len(available_clients) <= k: return list(available_clients)
    return list(rng.choice(available_clients, size=k, replace=False))

def always_available_oracle(all_clients, k, rng):
    return random_scheduler(all_clients, k, rng)

def no_availability_awareness(all_clients, k, rng):
    return random_scheduler(all_clients, k, rng)

def aws_scheduler(round_df, availability_df, persona_stats, last_seen, k, rng):
    cand = round_df[round_df.available == 1]["client_id"].drop_duplicates().tolist()
    if not cand: return []
    scores = []
    for cid in cand:
        p = availability_df[(availability_df.client_id==cid) & (availability_df.round_id==int(round_df.round_id.iloc[0]))]["availability_probability"]
        p = float(p.iloc[0]) if len(p) else 0.5
        fresh = 1.0 / (1 + last_seen.get(cid, 0))
        persona = round_df[round_df.client_id==cid]["persona"].mode().iloc[0]
        under = 1.0 / (1 + persona_stats.get(persona, 0))
        semantic_utility = min(1.0, round_df[round_df.client_id==cid]["text"].str.len().mean() / 200)
        corr_penalty = 0.1 * (persona_stats.get(persona, 0) > 0)
        s = 0.35*p + 0.2*fresh + 0.2*under + 0.25*semantic_utility - corr_penalty
        scores.append((cid, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c,_ in scores[:k]]
