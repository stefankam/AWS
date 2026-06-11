from __future__ import annotations
import argparse
import pandas as pd
import config
from models import build_model_and_tokenizer
from federated_train import run_federated
from baselines import (
    build_static_crosslm_prior_corpus,
    crosslm_teacher_student_baseline,
)
from evaluation import (
    evaluate_drift_completion_perplexity,
    evaluate_drift_target_perplexity,
    evaluate_perplexity_by_persona,
    evaluate_private_code_choice,
    evaluate_sentiment_accuracy_by_persona,
    evaluate_term_perplexity,
    summarize_fairness,
)
import visualizations as vz
from metrics import (
    detect_emerging_terms,
    detect_suppression_windows,
    measure_suppression_effect,
)

FEDERATED_METHODS = {"aws", "random", "oracle", "no_availability"}
CROSSLM_METHODS = {"centralized", "crosslm"}
SUPPORTED_METHODS = FEDERATED_METHODS | CROSSLM_METHODS

def _selected_client_count(metric_row: dict) -> int:
    if "selected_clients" in metric_row:
        try:
            return int(metric_row["selected_clients"])
        except (TypeError, ValueError):
            pass
    return int(metric_row.get("selected", 0) or 0)



def _selected_client_count(metric_row: dict) -> int:
    if "selected_clients" in metric_row:
        try:
            return int(metric_row["selected_clients"])
        except (TypeError, ValueError):
            pass
    return int(metric_row.get("selected", 0) or 0)


def _weighted_global_perplexity(per_persona_df: pd.DataFrame) -> float:
    if per_persona_df.empty:
        return float("nan")
    return float(
        (per_persona_df["perplexity"] * per_persona_df["num_eval_samples"]).sum()
        / max(per_persona_df["num_eval_samples"].sum(), 1)
    )


def _current_knowledge_eval_subset(round_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that specifically test up-to-date/semantic-drift knowledge."""
    if "drift_concept" not in round_df.columns:
        return round_df.iloc[0:0].copy()
    drift = round_df["drift_concept"].fillna("").astype(str).str.strip()
    return round_df[drift != ""].copy()


def run_method(
    method: str, df: pd.DataFrame, availability: pd.DataFrame, round_end_callback=None
):
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {sorted(SUPPORTED_METHODS)}"
        )

    model, tok = build_model_and_tokenizer(config.MODEL_NAME, config.USE_LORA)
    if method in CROSSLM_METHODS:
        rows = []
        # Static/stale LLM prior only.  CrossLM must not read merged/full client
        # rows, FinGPT current notes, round-specific semantic drift terms, or
        # local availability/persona metadata.
        llm_corpus = build_static_crosslm_prior_corpus(config)
        for r in range(config.NUM_ROUNDS):
            model, guidance_samples = crosslm_teacher_student_baseline(
                model, tok, llm_corpus, r, config, return_num_samples=True
            )


            row = {
                "method": method,
                "round": r,
                "selected": 0,
                "selected_clients": 0,
                "guidance_samples": guidance_samples,
                "guidance_round": int(guidance_samples > 0),
                "training_mode": "stale_teacher_student_guidance",
                "knowledge_source": "static_stale_llm_prior",
            }
            rows.append(row)
            if round_end_callback is not None:
                round_end_callback(model, tok, r, row)
        return model, tok, pd.DataFrame(rows), pd.DataFrame(columns=["method", "round", "selected_clients"])



    result = run_federated(method, model, tok, df, availability, config, round_end_callback=round_end_callback)
    if result is None:
        raise RuntimeError(f"Federated method {method!r} did not return model/metrics/selections")
    model, mdf, sdf = result
    if "selected_clients" not in mdf.columns and "selected" in mdf.columns:
        mdf["selected_clients"] = mdf["selected"]
    mdf["guidance_samples"] = 0
    mdf["guidance_round"] = 0
    mdf["training_mode"] = "federated_client_update"
    mdf["knowledge_source"] = "selected_up_to_date_client_streams"
    return model, tok, mdf, sdf



def evaluate_round_state(
    method: str,
    model,
    tok,
    round_id: int,
    metric_row: dict,
    merged: pd.DataFrame,
    all_pp: list[pd.DataFrame],
    all_fair: list[dict],
    lag_rows: list[dict],
    baseline_rows: list[dict],
    pending_terms: dict,
    adaptation_history: list[int],
):
    """Evaluate one method using the model state from the same training round."""
    ev = merged[merged.round_id == round_id]

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
    pp["method"] = method
    pp["round"] = round_id
    all_pp.append(pp)

    fair = summarize_fairness(pp)
    fair.update({"method": method, "round": round_id, "representation_imbalance": 0.0})
    all_fair.append(fair)

    terms = detect_emerging_terms(merged, list(range(round_id)), round_id)
    global_ppl = _weighted_global_perplexity(pp)



    current_ev = _current_knowledge_eval_subset(ev)
    current_knowledge_nll, current_knowledge_ppl, current_knowledge_samples, current_knowledge_tokens = (
        evaluate_drift_completion_perplexity(
            model,
            tok,
            current_ev,
            max_seq_length=config.MAX_SEQ_LENGTH,
            max_samples=config.EVAL_MAX_SAMPLES,
        )
    )
    current_target_nll, current_target_ppl, current_target_samples, current_target_tokens = (
        evaluate_drift_target_perplexity(
            model,
            tok,
            current_ev,
            max_seq_length=config.MAX_SEQ_LENGTH,
            max_samples=config.EVAL_MAX_SAMPLES,
        )
    )


    private_code_metrics = evaluate_private_code_choice(
        model,
        tok,
        current_ev,
        max_seq_length=config.MAX_SEQ_LENGTH,
        max_samples=config.EVAL_MAX_SAMPLES,
    )


    for term in terms[:5]:
        if term in pending_terms:
            continue
        first_ppl, first_samples = evaluate_term_perplexity(
            model,
            tok,
            ev["text"].astype(str).tolist(),
            term,
            max_seq_length=config.MAX_SEQ_LENGTH,
            batch_size=config.EVAL_BATCH_SIZE,
            max_samples=config.EVAL_MAX_SAMPLES,
        )
        pending_terms[term] = {
            "first_seen_round": round_id,
            "first_score": first_ppl,
            "threshold": first_ppl * config.ADAPTATION_PPL_IMPROVEMENT_RATIO if pd.notna(first_ppl) else float("nan"),
            "first_samples": first_samples,
            "resolved": False,
        }

    for term, state in pending_terms.items():
        if state["resolved"]:
            continue
        term_ppl, term_samples = evaluate_term_perplexity(
            model,
            tok,
            ev["text"].astype(str).tolist(),
            term,
            max_seq_length=config.MAX_SEQ_LENGTH,
            batch_size=config.EVAL_BATCH_SIZE,
            max_samples=config.EVAL_MAX_SAMPLES,
        )
        if term_samples == 0 or pd.isna(term_ppl) or pd.isna(state["threshold"]):
            continue
        if term_ppl <= state["threshold"]:
            lag = round_id - state["first_seen_round"]
            state["resolved"] = True
            adaptation_history.append(lag)
            lag_rows.append({
                "method": method,
                "term": term,
                "first_seen_round": state["first_seen_round"],
                "threshold_round": round_id,
                "adaptation_lag": lag,
                "first_term_perplexity": state["first_score"],
                "threshold_perplexity": state["threshold"],
                "observed_term_perplexity": term_ppl,
                "term_eval_samples": term_samples,
                "persona": "mixed",
                "region": "GLOBAL",
            })


    baseline_rows.append(
        {
            "method": method,
            "method_family": (
                "crosslm_teacher_student" if method in CROSSLM_METHODS else "federated"
            ),
            "round": round_id,
            "selected_clients": _selected_client_count(metric_row),
            "guidance_samples": int(metric_row.get("guidance_samples", 0) or 0),
            "num_eval_samples": int(pp["num_eval_samples"].sum()) if len(pp) else 0,
            "emerging_terms_count": len(terms),
            "global_perplexity": global_ppl,
            "current_knowledge_nll": current_knowledge_nll,
            "current_knowledge_perplexity": current_knowledge_ppl,
            "current_knowledge_samples": current_knowledge_samples,
            "current_knowledge_tokens": current_knowledge_tokens,
            "current_target_nll": current_target_nll,
            "current_target_perplexity": current_target_ppl,
            "current_target_samples": current_target_samples,
            "current_target_tokens": current_target_tokens,
            "private_code_accuracy": private_code_metrics["private_code_accuracy"],
            "private_code_margin": private_code_metrics["private_code_margin"],
            "private_code_correct_nll": private_code_metrics[
                "private_code_correct_nll"
            ],
            "private_code_samples": private_code_metrics["private_code_samples"],
            "knowledge_source": metric_row.get("knowledge_source", "unknown"),
            "training_mode": metric_row.get("training_mode", "unknown"),
            "worst_persona_perplexity": fair["worst_persona_perplexity"],
            "adaptation_lag_mean": (
                float(sum(adaptation_history) / len(adaptation_history))
                if adaptation_history
                else float("nan")
            ),
            "fairness_index": fair["jain_fairness"],
        }
    )




def summarize_current_knowledge_methods(
    baseline_df: pd.DataFrame, reference_method: str = "crosslm"
) -> pd.DataFrame:
    """Build an Option-E table: mean/final metrics, standard errors, and deltas."""
    metric_cols = [
        "global_perplexity",
        "current_knowledge_nll",
        "current_knowledge_perplexity",
        "current_target_nll",
        "current_target_perplexity",
        "private_code_accuracy",
        "private_code_margin",
        "private_code_correct_nll",
    ]
    rows = []
    if baseline_df.empty:
        return pd.DataFrame()

    for method, g in baseline_df.sort_values("round").groupby("method"):
        row = {"method": method, "num_rounds": int(g["round"].nunique())}
        for col in metric_cols:
            if col not in g.columns:
                continue
            vals = g[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            )
            row[f"{col}_se"] = (
                float(vals.std(ddof=1) / (len(vals) ** 0.5))
                if len(vals) > 1
                else float("nan")
            )
            row[f"{col}_final"] = float(vals.iloc[-1]) if len(vals) else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    ref = summary[summary["method"] == reference_method]
    if not ref.empty:
        ref_row = ref.iloc[0]
        higher_is_better = {"private_code_accuracy", "private_code_margin"}
        for col in metric_cols:
            for stat in ["mean", "final"]:
                key = f"{col}_{stat}"
                if key in summary.columns and key in ref_row:
                    if col in higher_is_better:
                        # Positive improvement means higher private-code accuracy/margin than CrossLM.
                        summary[f"{key}_improvement_vs_{reference_method}"] = (
                            summary[key] - ref_row[key]
                        )
                    else:
                        # Positive improvement means lower NLL/perplexity than CrossLM.
                        summary[f"{key}_improvement_vs_{reference_method}"] = (
                            ref_row[key] - summary[key]
                        )
    return summary




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

    for method in methods:
        adaptation_history = []
        pending_terms = {}
        method_checkpoint_dir = config.CHECKPOINT_DIR / method
        method_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def on_round_end(round_model, round_tokenizer, round_id, metric_row):
            round_dir = method_checkpoint_dir / f"round_{round_id:03d}"
            round_model.save_pretrained(round_dir)
            evaluate_round_state(
                method,
                round_model,
                round_tokenizer,
                round_id,
                metric_row,
                merged,
                all_pp,
                all_fair,
                lag_rows,
                baseline_rows,
                pending_terms,
                adaptation_history,
            )



        model, tok, mdf, sdf = run_method(method, merged, avail, round_end_callback=on_round_end)
        mdf.to_csv(config.METRICS_DIR / f"round_metrics_{method}.csv", index=False)
        sdf.to_csv(config.METRICS_DIR / f"selected_clients_{method}.csv", index=False)
        model.save_pretrained(method_checkpoint_dir / "final")

        for term, state in pending_terms.items():
            if state["resolved"]:
                continue
            lag_rows.append({
                "method": method,
                "term": term,
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


        method_pp = pd.concat([x for x in all_pp if not x.empty and (x["method"] == method).all()], ignore_index=True)
        if not method_pp.empty:
            for persona, g in avail_persona.groupby("persona"):
                region = g["region"].mode().iloc[0] if "region" in g.columns and len(g["region"].dropna()) else "GLOBAL"
                for window_start, window_end in detect_suppression_windows(avail_persona, persona, min_absent_rounds=2):
                    pre, mid, post, recovery_rounds = measure_suppression_effect(method_pp, persona, window_start, window_end)
                    suppress.append({
                        "method": method,
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
    perp = pd.concat(valid_pp, ignore_index=True) if valid_pp else pd.DataFrame(columns=["persona", "region", "nll", "perplexity", "num_eval_samples", "accuracy", "method", "round"])
    fairdf = pd.DataFrame(all_fair)
    lagdf = pd.DataFrame(lag_rows)
    suppressdf = pd.DataFrame(suppress, columns=["method", "persona", "region", "window_start", "window_end", "pre_window_metric", "during_window_metric", "post_window_metric", "recovery_rounds"])
    basedf = pd.DataFrame(baseline_rows)
    summarydf = summarize_current_knowledge_methods(basedf)

    perp.to_csv(config.METRICS_DIR / "per_persona_metrics.csv", index=False)
    fairdf.to_csv(config.METRICS_DIR / "fairness_metrics.csv", index=False)
    lagdf.to_csv(config.METRICS_DIR / "semantic_adaptation_lag.csv", index=False)
    suppressdf.to_csv(config.METRICS_DIR / "semantic_suppression.csv", index=False)
    basedf.to_csv(config.METRICS_DIR / "baseline_comparison.csv", index=False)
    summarydf.to_csv(config.METRICS_DIR / "current_knowledge_summary.csv", index=False)



    vz.plot_availability_heatmap(avail_persona, config.PLOTS_DIR)
    vz.plot_available_clients_per_round(avail_persona, config.PLOTS_DIR)
    vz.plot_semantic_drift_timeline(lagdf, config.PLOTS_DIR)
    vz.plot_per_persona_perplexity(perp, config.PLOTS_DIR)
    vz.plot_worst_persona_perplexity(fairdf, config.PLOTS_DIR)
    vz.plot_fairness_index(fairdf, config.PLOTS_DIR)
    vz.plot_adaptation_lag_by_method(lagdf, config.PLOTS_DIR)
    vz.plot_semantic_suppression_recovery(suppressdf, config.PLOTS_DIR)
    vz.plot_all_experiment_summaries(basedf, fairdf, perp, lagdf, config.PLOTS_DIR, summarydf=summarydf)


if __name__ == "__main__":
    main()
