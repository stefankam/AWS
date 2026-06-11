from __future__ import annotations
import pandas as pd


DEFAULT_SEMANTIC_DRIFT_CONCEPTS = [
    {
        "first_round": 8,
        "term": "spot bitcoin ETF flows",
        "snippet": "New concept: spot bitcoin ETF flows are changing crypto liquidity and cross-asset risk appetite.",
    },
    {
        "first_round": 16,
        "term": "MiCA stablecoin compliance",
        "snippet": "New concept: MiCA stablecoin compliance is affecting exchange listings, custody, and euro liquidity.",
    },
    {
        "first_round": 24,
        "term": "AI datacenter power bottleneck",
        "snippet": "New concept: AI datacenter power bottleneck risk is influencing utility demand and semiconductor capex.",
    },
    {
        "first_round": 32,
        "term": "private credit refinancing wall",
        "snippet": "New concept: private credit refinancing wall pressure may raise default risk for leveraged borrowers.",
    },
    {
        "first_round": 40,
        "term": "yen carry trade unwind",
        "snippet": "New concept: yen carry trade unwind risk can amplify FX volatility and global equity drawdowns.",
    },
]


def _resolve_first_round(first_round, num_rounds: int) -> int:
    if isinstance(first_round, float) and 0 <= first_round <= 1:
        return int(round(first_round * max(num_rounds - 1, 0)))
    return max(0, min(int(first_round), max(num_rounds - 1, 0)))


def _empty_drift_columns(out: pd.DataFrame) -> pd.DataFrame:
    out["drift_concept"] = ""
    out["drift_first_round"] = pd.NA
    out["drift_private_code"] = ""
    out["drift_answer"] = ""
    out["drift_sector"] = ""
    return out


def _append_semantic_drift(
    out: pd.DataFrame, num_rounds: int, concepts: list[dict]
) -> pd.DataFrame:
    if not concepts:
        return _empty_drift_columns(out)

    normalized = []
    for concept in concepts:
        first_round = _resolve_first_round(concept.get("first_round", 0), num_rounds)
        term = str(concept.get("term", concept.get("private_code", ""))).strip()
        if not term:
            continue
        private_code = str(concept.get("private_code", term)).strip()
        answer = str(concept.get("answer", concept.get("label", ""))).strip().lower()
        sector = str(concept.get("sector", "")).strip()
        snippet = str(
            concept.get(
                "snippet", f"New private market code: {term}. Signal label: {answer}."
            )
        ).strip()
        normalized.append(
            {
                "first_round": first_round,
                "term": term,
                "snippet": snippet,
                "private_code": private_code,
                "answer": answer,
                "sector": sector,
            }
        )

    if not normalized:
        return _empty_drift_columns(out)

    out = _empty_drift_columns(out)

    for idx, row in out.iterrows():
        active = [c for c in normalized if int(row["round_id"]) >= c["first_round"]]
        if not active:
            continue
        concept = active[idx % len(active)]
        out.at[idx, "text"] = f"{row['text']}\n\n{concept['snippet']}"
        if "prompt" in out.columns:
            out.at[idx, "prompt"] = (
                f"{row['prompt']}\n\nEmerging private market code to address: {concept['private_code']}"
            )
        out.at[idx, "drift_concept"] = concept["term"]
        out.at[idx, "drift_first_round"] = concept["first_round"]
        out.at[idx, "drift_private_code"] = concept["private_code"]
        out.at[idx, "drift_answer"] = concept["answer"]
        out.at[idx, "drift_sector"] = concept["sector"]

    return out




def add_temporal_rounds(
    df: pd.DataFrame,
    num_rounds: int = 50,
    semantic_drift_concepts: list[dict] | None = None,
) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["round_id"] = pd.qcut(
        out.index, q=num_rounds, labels=False, duplicates="drop"
    ).astype(int)
    out["round"] = out["round_id"]
    concepts = (
        DEFAULT_SEMANTIC_DRIFT_CONCEPTS
        if semantic_drift_concepts is None
        else semantic_drift_concepts
    )
    return _append_semantic_drift(out, num_rounds=num_rounds, concepts=concepts)
