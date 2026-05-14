"""Local client training helper."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader


def _tokenize_texts(tokenizer, texts, max_len):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")


def local_finetune(model, tokenizer, client_df, epochs: int, batch_size: int, lr: float, max_seq_length: int):
    if len(client_df) == 0:
        return model, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    enc = _tokenize_texts(tokenizer, client_df["text"].astype(str).tolist(), max_seq_length)
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
    return model, len(client_df)
