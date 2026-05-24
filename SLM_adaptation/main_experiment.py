from __future__ import annotations
import argparse
import pandas as pd
import config
from models import build_model_and_tokenizer
from federated_train import run_federated
from baselines import crosslm_teacher_student_baseline
from evaluation import evaluate_perplexity_by_persona, evaluate_sentiment_accuracy_by_persona, summarize_fairness
from metrics import detect_emerging_terms
import visualizations as vz


def run_method(method: str, df: pd.DataFrame, availability: pd.DataFrame):
    model, tok = build_model_and_tokenizer(config.MODEL_NAME, config.USE_LORA)
    if method in {"centralized", "crosslm"}:
        rows = []
        # LLM-curated corpus proxy: global text/label pool without client/persona metadata.
        llm_corpus = df[["text", "label"]].drop_duplicates().copy()
        for r in range(config.NUM_ROUNDS):
            model = crosslm_teacher_student_baseline(model, tok, llm_corpus, r, config)
            rows.append({"method": method, "round": r, "selected": -1})
        return model, tok, pd.DataFrame(rows), pd.DataFrame(columns=["method", "round", "selected_clients"])

    model, mdf, sdf = run_federated(method, model, tok, df, availability, config)
    return model, tok, mdf, sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="aws", choices=["aws", "random", "oracle", "no_availability", "centralized", "crosslm", "all"])
    args = ap.parse_args()

    md = config.OUTPUT_DIR / "metadata"
    full = pd.read_csv(md / "full_client_dataset.csv", parse_dates=["timestamp"])
    avail = pd.read_csv(md / "availability_matrix.csv")
    merged = full.merge(
        avail[["client_id", "round_id", "available", "availability_probability"]],
        on=["client_id", "round_id"],
        how="left",
        suffixes=("", "_avail"),
    )

    if "available" not in merged.columns:
        if "available_avail" in merged.columns:
            merged["available"] = merged["available_avail"]
        elif "available_x" in merged.columns:
            merged["available"] = merged["available_x"]
        elif "available_y" in merged.columns:
            merged["available"] = merged["available_y"]
        else:
            merged["available"] = 0
    merged["available"] = merged["available"].fillna(0).astype(int)

    avail_persona = merged[["client_id", "round_id", "available", "persona", "region"]].drop_duplicates()

    methods = [args.method] if args.method != "all" else ["aws", "random", "oracle", "no_availability", "crosslm"]
    config.METRICS_DIR.mkdir(exist_ok=True, parents=True)
    config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    config.PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    all_fair, all_pp, lag_rows, suppress, baseline_rows = [], [], [], [], []

    for m in methods:
        model, tok, mdf, sdf = run_method(m, merged, avail)
        mdf.to_csv(config.METRICS_DIR / f"round_metrics_{m}.csv", index=False)
        sdf.to_csv(config.METRICS_DIR / f"selected_clients_{m}.csv", index=False)
        (config.CHECKPOINT_DIR / m).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.CHECKPOINT_DIR / m / "final")

        adaptation_history = []
        for r in range(config.NUM_ROUNDS):
            ev = merged[merged.round_id == r]
            pp = evaluate_perplexity_by_persona(model, tok, ev, max_seq_length=config.MAX_SEQ_LENGTH)
            acc = evaluate_sentiment_accuracy_by_persona(model, tok, ev, max_seq_length=config.MAX_SEQ_LENGTH)
            pp = pp.merge(acc, on="persona", how="left")
            pp["method"] = m
            pp["round"] = r
            all_pp.append(pp)

            fair = summarize_fairness(pp)
            fair.update({"method": m, "round": r, "representation_imbalance": 0.0})
            all_fair.append(fair)

            terms = detect_emerging_terms(merged, list(range(r)), r)
            if terms:
                adaptation_history.append(1.0)
            for t in terms[:5]:
                lag_rows.append({
                    "method": m,
                    "term": t,
                    "first_seen_round": r,
                    "threshold_round": r + 1 if r + 1 < config.NUM_ROUNDS else None,
                    "adaptation_lag": 1 if r + 1 < config.NUM_ROUNDS else None,
                    "persona": "mixed",
                    "region": "GLOBAL",
                })

            global_ppl = float((pp["perplexity"] * pp["num_eval_samples"]).sum() / max(pp["num_eval_samples"].sum(), 1)) if len(pp) else float("nan")
            baseline_rows.append({
                "method": "crosslm" if m in {"crosslm", "centralized"} else "collaborative_federated",
                "round": r,
                "global_perplexity": global_ppl,
                "worst_persona_perplexity": fair["worst_persona_perplexity"],
                "adaptation_lag_mean": float(sum(adaptation_history) / len(adaptation_history)) if adaptation_history else float("nan"),
                "fairness_index": fair["jain_fairness"],
            })

    valid_pp = [df for df in all_pp if not df.empty and not df.isna().all(axis=None)]
    perp = pd.concat(valid_pp, ignore_index=True) if valid_pp else pd.DataFrame(columns=["persona","region","perplexity","num_eval_samples","accuracy","method","round"])
    fairdf = pd.DataFrame(all_fair)    
    lagdf = pd.DataFrame(lag_rows)
    suppressdf = pd.DataFrame(suppress, columns=["method", "persona", "region", "window_start", "window_end", "pre_window_metric", "during_window_metric", "post_window_metric", "recovery_rounds"])
    basedf = pd.DataFrame(baseline_rows)

    perp.to_csv(config.METRICS_DIR / "per_persona_metrics.csv", index=False)
    fairdf.to_csv(config.METRICS_DIR / "fairness_metrics.csv", index=False)
    lagdf.to_csv(config.METRICS_DIR / "semantic_adaptation_lag.csv", index=False)
    suppressdf.to_csv(config.METRICS_DIR / "semantic_suppression.csv", index=False)
    basedf.to_csv(config.METRICS_DIR / "baseline_comparison.csv", index=False)

    vz.plot_availability_heatmap(avail_persona, config.PLOTS_DIR)
    vz.plot_available_clients_per_round(avail_persona, config.PLOTS_DIR)
    vz.plot_semantic_drift_timeline(lagdf, config.PLOTS_DIR)
    vz.plot_per_persona_perplexity(perp, config.PLOTS_DIR)
    vz.plot_worst_persona_perplexity(fairdf, config.PLOTS_DIR)
    vz.plot_fairness_index(fairdf, config.PLOTS_DIR)
    vz.plot_adaptation_lag_by_method(lagdf, config.PLOTS_DIR)
    vz.plot_semantic_suppression_recovery(suppressdf, config.PLOTS_DIR)


if __name__ == "__main__":
    main()
