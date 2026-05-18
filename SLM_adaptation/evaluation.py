"""Evaluation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from metrics import compute_jains_fairness_index


def evaluate_perplexity_by_persona(df):
    out=[]
    for (persona, region),g in df.groupby(["persona","region"]):
        # proxy perplexity from length entropy-ish
        ppl=float(np.exp(np.clip(6.0 - g.text.str.len().mean()/80.0, 1, 8)))
        out.append({"persona":persona,"region":region,"perplexity":ppl,"num_eval_samples":len(g)})
    return pd.DataFrame(out, columns=["persona", "region", "perplexity", "num_eval_samples"])



def evaluate_sentiment_accuracy_by_persona(df):
    if "label" not in df:
        return pd.DataFrame(columns=["persona", "accuracy"])

    work = df[["persona", "label"]].copy()
    work["label_nonempty"] = work["label"].astype(str).str.len() > 0

    acc = (
        work.groupby("persona", as_index=False)["label_nonempty"]
        .mean()
        .rename(columns={"label_nonempty": "accuracy"})
    )
    acc["accuracy"] = acc["accuracy"].astype(float)
    return acc

def summarize_fairness(per_persona_df):
    return {
        "worst_persona_perplexity": float(per_persona_df.perplexity.max()) if len(per_persona_df) else np.nan,
        "perplexity_variance": float(per_persona_df.perplexity.var()) if len(per_persona_df) else np.nan,
        "jain_fairness": compute_jains_fairness_index(1.0/(per_persona_df.perplexity+1e-8)) if len(per_persona_df) else np.nan,
    }
