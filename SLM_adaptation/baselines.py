"""Centralized baselines, including a CrossLM-inspired teacher-student path."""
from __future__ import annotations

import copy
import pandas as pd

from local_train import local_finetune


_LOCAL_ONLY_COLUMNS = {
    "client_id",
    "persona",
    "region",
    "timezone_group",
    "available",
    "availability_probability",
}

def _guidance_every(cfg) -> int:
    return max(1, int(getattr(cfg, "CROSSLM_GUIDANCE_EVERY", getattr(cfg, "CENTRAL_RETRAIN_EVERY", 5))))


def _max_guidance_samples(cfg) -> int | None:
    value = getattr(cfg, "CROSSLM_MAX_GUIDANCE_SAMPLES_PER_ROUND", None)
    return int(value) if value is not None else None


def build_crosslm_guidance_batch(llm_curated_df: pd.DataFrame, round_id: int, cfg) -> pd.DataFrame:
    """Build one periodic LLM-teacher guidance batch for the student SLM.

    The batch intentionally excludes local-only fields such as client, persona,
    region, and availability. If a public/LLM corpus carries ``round_id``, the
    teacher exposes only samples available up to the current guidance round.
    """
    if round_id % _guidance_every(cfg) != 0 or llm_curated_df.empty:
        return pd.DataFrame(columns=["text", "label"])

    public_cols = [c for c in llm_curated_df.columns if c not in _LOCAL_ONLY_COLUMNS]
    work = llm_curated_df[public_cols].copy()
    if "round_id" in work.columns:
        work = work[work["round_id"] <= round_id]
    if work.empty or "text" not in work.columns:
        return pd.DataFrame(columns=["text", "label"])

    work = work.dropna(subset=["text"]).drop_duplicates().copy()
    max_samples = _max_guidance_samples(cfg)
    if max_samples is not None and len(work) > max_samples:
        work = work.sample(n=max_samples, random_state=getattr(cfg, "SEED", 42) + round_id)
    else:
        work = work.sample(frac=1.0, random_state=getattr(cfg, "SEED", 42) + round_id)

    rows = []
    for _, row in work.iterrows():
        prompt = str(row.get("prompt", "Generate a concise financial assistant note.")).strip()
        teacher_note = str(row["text"]).strip()
        label = str(row.get("label", "neutral")).strip() or "neutral"
        topic = str(row.get("topic", "general financial markets")).strip()
        guidance_text = (
            f"CrossLM teacher guidance round {round_id}.\n"
            f"Public topic: {topic}\n"
            f"Teacher prompt: {prompt}\n"
            f"Teacher note: {teacher_note}\n"
            f"Student SLM target: {teacher_note}\n"
            f"Sentiment label: [{label}]"
        )
        rows.append({"text": guidance_text, "label": label})

    return pd.DataFrame(rows, columns=["text", "label"])


def crosslm_teacher_student_baseline(
    global_model,
    tokenizer,
    llm_curated_df: pd.DataFrame,
    round_id: int,
    cfg,
    return_num_samples: bool = False,
):

    """CrossLM-inspired periodic teacher-student baseline.

    Baseline semantics for this repository:
    - A central LLM teacher periodically creates guidance examples.
    - The student SLM adapts round-wise on those teacher-guided examples.
    - Client-local streams, persona identifiers, and availability are not used.
    - No FedAvg or client replica aggregation is performed.
    """


    guidance_df = build_crosslm_guidance_batch(llm_curated_df, round_id, cfg)
    if guidance_df.empty:
        return (global_model, 0) if return_num_samples else global_model

    server_model = copy.deepcopy(global_model)
    server_model, n = local_finetune(
        model=server_model,
        tokenizer=tokenizer,
        client_df=guidance_df,
        epochs=getattr(cfg, "CROSSLM_GUIDANCE_EPOCHS", cfg.LOCAL_EPOCHS),
        batch_size=getattr(cfg, "CROSSLM_GUIDANCE_BATCH_SIZE", cfg.LOCAL_BATCH_SIZE),
        lr=getattr(cfg, "CROSSLM_GUIDANCE_LR", cfg.LEARNING_RATE),
        max_seq_length=cfg.MAX_SEQ_LENGTH,
    )
    return (server_model, n) if return_num_samples else server_model

def central_llm_guidance_baseline(*args, **kwargs):
    """Backward-compatible alias for the old baseline name."""
    return crosslm_teacher_student_baseline(*args, **kwargs)
