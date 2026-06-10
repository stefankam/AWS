"""Local client training helper."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader


def _tokenize_texts(tokenizer, texts, max_len):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")


def local_finetune(
    model,
    tokenizer,
    client_df,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_length: int,
    semantic_drift_oversample: int = 1,
):
    if len(client_df) == 0:
        return model, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    texts = client_df["text"].astype(str).tolist()
    if "drift_concept" in client_df.columns:
        drift_mask = client_df["drift_concept"].fillna("").astype(str).str.strip() != ""
        drift_texts = client_df.loc[drift_mask, "text"].astype(str).tolist()
        # The experiment's freshness hypothesis depends on local SLMs learning
        # client-only semantic-drift snippets.  Oversample those examples during
        # local training so the new concepts are not washed out by generic
        # finance-language tokens.
        repeat = max(1, int(semantic_drift_oversample))
        texts.extend(drift_texts * (repeat - 1))
    enc = _tokenize_texts(tokenizer, texts, max_seq_length)
    enc["labels"] = enc["input_ids"].clone()
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for b in dl:
            b = [x.to(device) for x in b]
            out = model(input_ids=b[0], attention_mask=b[1], labels=b[2])
            out.loss.backward()
            opt.step(); opt.zero_grad()
    return model, len(texts)
