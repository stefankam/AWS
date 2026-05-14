"""Metrics for adaptation lag, fairness, and suppression."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd


def extract_round_terms(df, min_frequency=3):
    toks = " ".join(df["text"].astype(str).str.lower()).split()
    c = Counter(toks)
    return {t for t,v in c.items() if v>=min_frequency and len(t)>3}

def detect_emerging_terms(df, previous_rounds, current_round):
    prev = extract_round_terms(df[df.round_id.isin(previous_rounds)]) if previous_rounds else set()
    cur = extract_round_terms(df[df.round_id==current_round])
    return sorted(cur - prev)

def compute_term_perplexity(model, tokenizer, eval_texts, target_terms):
    # proxy: negative log frequency for target term coverage
    txt = " ".join(eval_texts).lower()
    out = {}
    for t in target_terms:
        f = txt.count(t.lower()) + 1
        out[t] = float(np.log(1 + 1/f) + 1.0)
    return out

def compute_adaptation_lag(metric_history, term, threshold):
    first = metric_history[0]["round"] if metric_history else None
    hit = next((x["round"] for x in metric_history if x["term"]==term and x["score"]<=threshold), None)
    if first is None or hit is None: return None
    return int(hit-first)

def compute_worst_persona_perplexity(df): return float(df["perplexity"].max()) if len(df) else np.nan
def compute_persona_perplexity_variance(df): return float(df["perplexity"].var()) if len(df) else np.nan
def compute_jains_fairness_index(vals):
    x=np.array(vals,dtype=float); x=x[~np.isnan(x)]
    if len(x)==0 or (x**2).sum()==0:return np.nan
    return float((x.sum()**2)/(len(x)*(x**2).sum()))
def compute_cumulative_contribution(sel_hist): return dict(Counter(sel_hist))
def compute_representation_imbalance(counts):
    arr=np.array(list(counts.values()),dtype=float)
    return float((arr.max()-arr.min())/(arr.mean()+1e-8)) if len(arr) else np.nan

def detect_suppression_windows(availability_df, persona, min_absent_rounds):
    p = availability_df[availability_df.persona==persona].groupby("round_id")["available"].sum().reset_index()
    windows=[]; start=None
    for _,r in p.iterrows():
        if r.available==0 and start is None: start=int(r.round_id)
        if r.available>0 and start is not None:
            if int(r.round_id)-start>=min_absent_rounds: windows.append((start,int(r.round_id)-1))
            start=None
    return windows

def measure_suppression_effect(metric_history, persona, window_start, window_end):
    d=metric_history[metric_history.persona==persona]
    pre=d[d.round<window_start]["perplexity"].mean(); mid=d[(d.round>=window_start)&(d.round<=window_end)]["perplexity"].mean(); post=d[d.round>window_end]["perplexity"].mean()
    rec = int((d[d.round>window_end]["perplexity"].cummin().idxmin()-window_end)) if len(d[d.round>window_end]) else np.nan
    return pre,mid,post,rec

def compare_recovery_after_suppression(methods): return pd.DataFrame(methods)
