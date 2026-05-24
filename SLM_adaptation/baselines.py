"""Centralized baselines, including a CrossLM-inspired teacher-student path."""
from __future__ import annotations

import copy
import pandas as pd

from local_train import local_finetune

def crosslm_teacher_student_baseline(
    global_model,
    tokenizer,
    llm_curated_df: pd.DataFrame,
    round_id: int,
    cfg,
):


    """CrossLM-inspired non-federated baseline.
    Baseline semantics for this repository:
    - Student SLM is trained centrally from an LLM-curated corpus.
    - No client-local stream is consumed during training.
    - No persona/client identifiers are used.
    - Model is trained once and then reused as a fixed baseline.
    """
    if round_id != 0 or llm_curated_df.empty:
        return global_model

    server_model = copy.deepcopy(global_model)
    train_df = llm_curated_df[["text", "label"]].dropna(subset=["text"]).copy()
    train_df = train_df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    server_model, _ = local_finetune(
        model=server_model,
        tokenizer=tokenizer,
        client_df=train_df,
        epochs=cfg.LOCAL_EPOCHS,
        batch_size=cfg.LOCAL_BATCH_SIZE,
        lr=cfg.LEARNING_RATE,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
    )
    return server_model


def central_llm_guidance_baseline(*args, **kwargs):
    """Backward-compatible alias for the old baseline name."""
    return crosslm_teacher_student_baseline(*args, **kwargs)
