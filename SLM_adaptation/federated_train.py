"""Federated training loop."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from aggregation import aggregate_model_updates
from local_train import local_finetune
from schedulers import random_scheduler, aws_scheduler, always_available_oracle, no_availability_awareness


def run_federated(method, global_model, tokenizer, full_df, availability_df, cfg):
    rng=np.random.default_rng(cfg.SEED)
    metrics=[]; selections=[]; persona_stats={}; last_seen={}
    all_clients=sorted(full_df.client_id.unique().tolist())
    for r in range(cfg.NUM_ROUNDS):
        round_df=full_df[full_df.round_id==r].copy()
        avail=round_df[round_df.available==1].client_id.drop_duplicates().tolist()
        if method=="random": sel=random_scheduler(avail, cfg.CLIENTS_PER_ROUND, rng)
        elif method=="aws": sel=aws_scheduler(round_df, availability_df, persona_stats, last_seen, cfg.CLIENTS_PER_ROUND, rng)
        elif method=="oracle": sel=always_available_oracle(all_clients, cfg.CLIENTS_PER_ROUND, rng)
        elif method=="no_availability": sel=no_availability_awareness(all_clients, cfg.CLIENTS_PER_ROUND, rng)
        else: sel=[]
        selections.append({"method":method,"round":r,"selected_clients":";".join(map(str,sel))})
        if not sel:
            metrics.append({"method":method,"round":r,"selected":0})
            continue
        locals=[]; weights=[]
        for cid in sel:
            cdf=round_df[round_df.client_id==cid]
            if len(cdf)==0: continue
            lm=copy.deepcopy(global_model)
            lm,n=local_finetune(lm, tokenizer, cdf, cfg.LOCAL_EPOCHS, cfg.LOCAL_BATCH_SIZE, cfg.LEARNING_RATE, cfg.MAX_SEQ_LENGTH)
            locals.append(lm); weights.append(n)
            persona=cdf.persona.mode().iloc[0]; persona_stats[persona]=persona_stats.get(persona,0)+1; last_seen[cid]=0
        global_model=aggregate_model_updates(global_model, locals, weights)
        for cid in all_clients:
            last_seen[cid]=last_seen.get(cid,0)+1
        metrics.append({"method":method,"round":r,"selected":len(sel)})
    return global_model, pd.DataFrame(metrics), pd.DataFrame(selections)
