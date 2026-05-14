"""Centralized CrossLM-style proxy baselines."""
from __future__ import annotations

import copy
import pandas as pd

from local_train import local_finetune


def central_llm_guidance_baseline(
    global_model,
    tokenizer,
    visible_df: pd.DataFrame,
    retrain_every: int,
    round_id: int,
    cfg,
):
    """CrossLM-style proxy: server-side post-training from client SLM data streams.

    Key behavior:
    - One central server model is updated (not FedAvg over local model replicas).
    - All client streams observed up to current round contribute when retraining is triggered.
    - Client availability is ignored for this baseline (upper-bound centralized guidance).
    - No per-client personalized model is preserved in this path.
    """
    if round_id % retrain_every != 0:
        return global_model

    if visible_df.empty:
        return global_model

    server_model = copy.deepcopy(global_model)

    # Simulate client SLM knowledge transfer to central model by iterating over
    # per-client corpora (all clients, all visible rounds) and post-training on server.
    for _, client_df in visible_df.groupby("client_id"):
        if client_df.empty:
            continue
        server_model, _ = local_finetune(
            model=server_model,
            tokenizer=tokenizer,
            client_df=client_df[["text", "label"]].copy(),
            epochs=cfg.LOCAL_EPOCHS,
            batch_size=cfg.LOCAL_BATCH_SIZE,
            lr=cfg.LEARNING_RATE,
            max_seq_length=cfg.MAX_SEQ_LENGTH,
        )

    return server_model
