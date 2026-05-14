"""FedAvg aggregation for full or adapter parameters."""
from __future__ import annotations
import copy
import torch


def aggregate_model_updates(global_model, local_models, client_weights):
    if not local_models:
        return global_model
    total = sum(client_weights) if sum(client_weights) > 0 else 1
    new_model = copy.deepcopy(global_model)
    state = new_model.state_dict()
    keys = [k for k in state if any(x in k for x in ["lora", "adapter"]) ] or [k for k,v in state.items() if v.dtype.is_floating_point]
    for k in keys:
        agg = None
        for m, w in zip(local_models, client_weights):
            t = m.state_dict()[k].detach().float() * (w / total)
            agg = t if agg is None else agg + t
        state[k] = agg.to(state[k].dtype)
    new_model.load_state_dict(state, strict=False)
    return new_model
