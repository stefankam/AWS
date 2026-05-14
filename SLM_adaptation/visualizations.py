"""Paper-ready matplotlib visualizations (PNG+PDF)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def _save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def plot_availability_heatmap(availability_df, out):
    p=availability_df.pivot(index='client_id', columns='round_id', values='available').fillna(0)
    fig=plt.figure(figsize=(11,5)); plt.imshow(p.values, aspect='auto'); plt.xlabel('round'); plt.ylabel('client'); plt.title('Availability heatmap'); plt.colorbar(); _save(fig,out/'availability_heatmap')

def plot_available_clients_per_round(availability_df, out):
    c=availability_df.groupby('round_id')['available'].sum(); fig=plt.figure(figsize=(10,4)); plt.plot(c.index,c.values,marker='o'); plt.title('Available clients per round'); plt.xlabel('round'); plt.ylabel('clients'); _save(fig,out/'available_clients_per_round')

def plot_semantic_drift_timeline(lag_df, out):
    fig=plt.figure(figsize=(10,5));
    if len(lag_df):
        y=range(len(lag_df)); plt.scatter(lag_df.first_seen_round,y,label='first_seen'); plt.scatter(lag_df.threshold_round.fillna(-1),y,label='threshold');
        plt.yticks(list(y), lag_df.term)
    plt.legend(); plt.title('Semantic drift timeline'); plt.xlabel('round'); _save(fig,out/'semantic_drift_timeline')

def plot_per_persona_perplexity(pp, out):
    fig=plt.figure(figsize=(10,5));
    for p,g in pp.groupby('persona'): plt.plot(g['round'],g['perplexity'],label=p)
    plt.legend(ncol=2,fontsize=8); plt.title('Per-persona perplexity'); _save(fig,out/'per_persona_perplexity')

def plot_worst_persona_perplexity(fair,out):
    fig=plt.figure(figsize=(9,4));
    for m,g in fair.groupby('method'): plt.plot(g['round'],g['worst_persona_perplexity'],label=m)
    plt.legend(); plt.title('Worst-persona perplexity'); _save(fig,out/'worst_persona_perplexity')

def plot_fairness_index(fair,out):
    fig=plt.figure(figsize=(9,4));
    for m,g in fair.groupby('method'): plt.plot(g['round'],g['jain_fairness'],label=m)
    plt.legend(); plt.title('Fairness index'); _save(fig,out/'fairness_index')

def plot_adaptation_lag_by_method(lag,out):
    fig=plt.figure(figsize=(9,4));
    if len(lag):
        m=lag.groupby('method')['adaptation_lag'].mean(); plt.bar(m.index,m.values)
    plt.title('Adaptation lag by method'); _save(fig,out/'adaptation_lag_by_method')

def plot_semantic_suppression_recovery(ss,out):
    fig=plt.figure(figsize=(9,4));
    if len(ss):
        plt.bar(range(len(ss)), ss['recovery_rounds'].fillna(0)); plt.xticks(range(len(ss)), ss['persona'], rotation=30)
    plt.title('Temporal semantic suppression recovery'); _save(fig,out/'semantic_suppression_recovery')
