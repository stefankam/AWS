"""Paper-ready matplotlib visualizations (PNG+PDF)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def _save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def _plot_lines(df, x, y, group, title, ylabel, out_path):
    fig = plt.figure(figsize=(10, 5))
    if len(df) and y in df.columns:
        for name, g in df.dropna(subset=[y]).groupby(group):
            plt.plot(g[x], g[y], marker='o', label=str(name))
        plt.legend(ncol=2, fontsize=8)
    plt.title(title)
    plt.xlabel(x.replace('_', ' '))
    plt.ylabel(ylabel)
    _save(fig, out_path)


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
    if len(lag_df):
        plt.legend()
    plt.title('Semantic drift timeline'); plt.xlabel('round'); _save(fig,out/'semantic_drift_timeline')

def plot_per_persona_perplexity(pp, out):
    fig=plt.figure(figsize=(11,6));
    if len(pp):
        for (method, persona), g in pp.groupby(['method', 'persona']):
            plt.plot(g['round'], g['perplexity'], label=f'{method}:{persona}')
        plt.legend(ncol=2,fontsize=7)
    plt.title('Per-persona perplexity by method'); plt.xlabel('round'); plt.ylabel('perplexity'); _save(fig,out/'per_persona_perplexity')

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
    plt.title('Adaptation lag by method'); plt.ylabel('mean lag (rounds)'); _save(fig,out/'adaptation_lag_by_method')


def plot_semantic_suppression_recovery(ss,out):
    fig=plt.figure(figsize=(9,4));
    if len(ss):
        plt.bar(range(len(ss)), ss['recovery_rounds'].fillna(0)); plt.xticks(range(len(ss)), ss['persona'], rotation=30)
    plt.title('Temporal semantic suppression recovery'); _save(fig,out/'semantic_suppression_recovery')


def plot_global_perplexity_by_method(baseline_df, out):
    _plot_lines(baseline_df, 'round', 'global_perplexity', 'method', 'Global perplexity by experiment', 'perplexity', out/'global_perplexity_by_method')


def plot_selected_clients_by_method(baseline_df, out):
    _plot_lines(baseline_df, 'round', 'selected_clients', 'method', 'Selected clients by experiment', 'clients selected', out/'selected_clients_by_method')


def plot_emerging_terms_by_method(baseline_df, out):
    _plot_lines(baseline_df, 'round', 'emerging_terms_count', 'method', 'Emerging terms by experiment', 'terms detected', out/'emerging_terms_by_method')


def plot_sentiment_accuracy_by_method(pp, out):
    fig = plt.figure(figsize=(10, 5))
    if len(pp) and 'accuracy' in pp.columns:
        acc = pp.dropna(subset=['accuracy']).groupby(['method', 'round'], as_index=False)['accuracy'].mean()
        for method, g in acc.groupby('method'):
            plt.plot(g['round'], g['accuracy'], marker='o', label=str(method))
        plt.legend(ncol=2, fontsize=8)
    plt.title('Mean sentiment accuracy by experiment')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    _save(fig, out/'sentiment_accuracy_by_method')


def plot_final_metric_comparison(baseline_df, out):
    fig = plt.figure(figsize=(10, 5))
    if len(baseline_df):
        last = baseline_df.sort_values('round').groupby('method', as_index=False).tail(1)
        x = range(len(last))
        plt.bar(x, last['global_perplexity'])
        plt.xticks(x, last['method'], rotation=30, ha='right')
    plt.title('Final-round global perplexity comparison')
    plt.ylabel('perplexity')
    _save(fig, out/'final_global_perplexity_comparison')


def plot_all_experiment_summaries(baseline_df, fair_df, per_persona_df, lag_df, out):
    """Create plots that compare every experiment method included in a run."""
    out = Path(out)
    plot_global_perplexity_by_method(baseline_df, out)
    plot_selected_clients_by_method(baseline_df, out)
    plot_emerging_terms_by_method(baseline_df, out)
    plot_sentiment_accuracy_by_method(per_persona_df, out)
    plot_final_metric_comparison(baseline_df, out)
    plot_worst_persona_perplexity(fair_df, out)
    plot_fairness_index(fair_df, out)
    plot_adaptation_lag_by_method(lag_df, out)
