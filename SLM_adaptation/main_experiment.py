from __future__ import annotations
import argparse
import pandas as pd
import config
from models import build_model_and_tokenizer
from federated_train import run_federated
from baselines import crosslm_teacher_student_baseline
from evaluation import evaluate_perplexity_by_persona, evaluate_sentiment_accuracy_by_persona, evaluate_term_perplexity, summarize_fairness
import visualizations as vz
from metrics import detect_emerging_terms, detect_suppression_windows, measure_suppression_effect


FEDERATED_METHODS = {"aws", "random", "oracle", "no_availability"}
CROSSLM_METHODS = {"centralized", "crosslm"}
SUPPORTED_METHODS = FEDERATED_METHODS | CROSSLM_METHODS



def run_method(method: str, df: pd.DataFrame, availability: pd.DataFrame):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(SUPPORTED_METHODS)}")

    model, tok = build_model_and_tokenizer(config.MODEL_NAME, config.USE_LORA)
    if method in CROSSLM_METHODS:
        rows = []
        # LLM-curated/public corpus proxy; strip client/persona/availability metadata.
        local_only = {"client_id", "persona", "region", "timezone_group", "available", "availability_probability"}
        public_cols = [c for c in df.columns if c not in local_only]
        llm_corpus = df[public_cols].drop_duplicates().copy()
        for r in range(config.NUM_ROUNDS):
            model, guidance_samples = crosslm_teacher_student_baseline(
                model, tok, llm_corpus, r, config, return_num_samples=True
            )
            rows.append({
                "method": method,
                "round": r,
                "selected": 0,
                "selected_clients": 0,
                "guidance_samples": guidance_samples,
                "guidance_round": int(guidance_samples > 0),
                "training_mode": "teacher_student_guidance",
            })
        return model, tok, pd.DataFrame(rows), pd.DataFrame(columns=["method", "round", "selected_clients"])

    result = run_federated(method, model, tok, df, availability, config)
    if result is None:
        raise RuntimeError(f"Federated method {method!r} did not return model/metrics/selections")
    model, mdf, sdf = result
    if "selected_clients" not in mdf.columns and "selected" in mdf.columns:
        mdf["selected_clients"] = mdf["selected"]
    mdf["guidance_samples"] = 0
    mdf["guidance_round"] = 0
    mdf["training_mode"] = "federated_client_update"
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
        pending_terms = {}
        for r in range(config.NUM_ROUNDS):
            ev = merged[merged.round_id == r]

            pp = evaluate_perplexity_by_persona(
                model,
                tok,
                ev,
                max_seq_length=config.MAX_SEQ_LENGTH,
                eval_batch_size=config.EVAL_BATCH_SIZE,
                eval_max_samples=config.EVAL_MAX_SAMPLES,
            )
            acc = evaluate_sentiment_accuracy_by_persona(
                model,
                tok,
                ev,
                max_seq_length=config.MAX_SEQ_LENGTH,
                eval_max_samples=config.EVAL_MAX_SAMPLES,
            )
            pp = pp.merge(acc, on="persona", how="left")
            pp["method"] = m
            pp["round"] = r
            all_pp.append(pp)

            fair = summarize_fairness(pp)
            fair.update({"method": m, "round": r, "representation_imbalance": 0.0})
            all_fair.append(fair)

            terms = detect_emerging_terms(merged, list(range(r)), r)
            global_ppl = float((pp["perplexity"] * pp["num_eval_samples"]).sum() / max(pp["num_eval_samples"].sum(), 1)) if len(pp) else float("nan")

            for t in terms[:5]:
                if t in pending_terms:
                    continue
                first_ppl, first_samples = evaluate_term_perplexity(
                    model,
                    tok,
                    ev["text"].astype(str).tolist(),
                    t,
                    max_seq_length=config.MAX_SEQ_LENGTH,
                    batch_size=config.EVAL_BATCH_SIZE,
                    max_samples=config.EVAL_MAX_SAMPLES,
                )
                pending_terms[t] = {
                    "first_seen_round": r,
                    "first_score": first_ppl,
                    "threshold": first_ppl * config.ADAPTATION_PPL_IMPROVEMENT_RATIO if pd.notna(first_ppl) else float("nan"),
                    "first_samples": first_samples,
                    "resolved": False,
                }

            for t, state in pending_terms.items():
                if state["resolved"]:
                    continue
                term_ppl, term_samples = evaluate_term_perplexity(
                    model,
                    tok,
                    ev["text"].astype(str).tolist(),
                    t,
                    max_seq_length=config.MAX_SEQ_LENGTH,
                    batch_size=config.EVAL_BATCH_SIZE,
                    max_samples=config.EVAL_MAX_SAMPLES,
                )
                if term_samples == 0 or pd.isna(term_ppl) or pd.isna(state["threshold"]):
                    continue
                if term_ppl <= state["threshold"]:
                    lag = r - state["first_seen_round"]
                    state["resolved"] = True
                    adaptation_history.append(lag)
                    lag_rows.append({
                        "method": m,
                        "term": t,
                        "first_seen_round": state["first_seen_round"],
                        "threshold_round": r,
                        "adaptation_lag": lag,
                        "first_term_perplexity": state["first_score"],
                        "threshold_perplexity": state["threshold"],
                        "observed_term_perplexity": term_ppl,
                        "term_eval_samples": term_samples,
                        "persona": "mixed",
                        "region": "GLOBAL",
                    })
            selected_row = mdf[mdf["round"] == r]
            if not selected_row.empty and "selected_clients" in selected_row:
                selected_clients = int(selected_row["selected_clients"].iloc[0])
            elif not selected_row.empty and "selected" in selected_row:
                selected_clients = int(selected_row["selected"].iloc[0])
            else:
                selected_clients = 0
            guidance_samples = int(selected_row["guidance_samples"].iloc[0]) if not selected_row.empty and "guidance_samples" in selected_row else 0
            baseline_rows.append({
                "method": m,
                "method_family": "crosslm_teacher_student" if m in {"crosslm", "centralized"} else "federated",
                "round": r,
                "selected_clients": selected_clients,
                "guidance_samples": guidance_samples,
                "num_eval_samples": int(pp["num_eval_samples"].sum()) if len(pp) else 0,
                "emerging_terms_count": len(terms),
                "global_perplexity": global_ppl,
                "worst_persona_perplexity": fair["worst_persona_perplexity"],
                "adaptation_lag_mean": float(sum(adaptation_history) / len(adaptation_history)) if adaptation_history else float("nan"),
                "fairness_index": fair["jain_fairness"],
            })

        for t, state in pending_terms.items():
            if state["resolved"]:
                continue
            lag_rows.append({
                "method": m,
                "term": t,
                "first_seen_round": state["first_seen_round"],
                "threshold_round": None,
                "adaptation_lag": None,
                "first_term_perplexity": state["first_score"],
                "threshold_perplexity": state["threshold"],
                "observed_term_perplexity": None,
                "term_eval_samples": 0,
                "persona": "mixed",
                "region": "GLOBAL",
            })

        method_pp = pd.concat([x for x in all_pp if not x.empty and (x["method"] == m).all()], ignore_index=True)
        if not method_pp.empty:
            for persona, g in avail_persona.groupby("persona"):
                region = g["region"].mode().iloc[0] if "region" in g.columns and len(g["region"].dropna()) else "GLOBAL"
                for window_start, window_end in detect_suppression_windows(avail_persona, persona, min_absent_rounds=2):
                    pre, mid, post, recovery_rounds = measure_suppression_effect(method_pp, persona, window_start, window_end)
                    suppress.append({
                        "method": m,
                        "persona": persona,
                        "region": region,
                        "window_start": window_start,
                        "window_end": window_end,
                        "pre_window_metric": pre,
                        "during_window_metric": mid,
                        "post_window_metric": post,
                        "recovery_rounds": recovery_rounds,
                    })

    valid_pp = [df for df in all_pp if not df.empty and not df.isna().all(axis=None)]
    perp = pd.concat(valid_pp, ignore_index=True) if valid_pp else pd.DataFrame(columns=["persona","region","nll","perplexity","num_eval_samples","accuracy","method","round"])
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
    vz.plot_all_experiment_summaries(basedf, fairdf, perp, lagdf, config.PLOTS_DIR)

if __name__ == "__main__":
    main()
