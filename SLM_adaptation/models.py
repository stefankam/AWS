"""Model/tokenizer builders with optional LoRA wrapping."""
from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def build_model_and_tokenizer(model_name: str, use_lora: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if use_lora:
        lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    return model, tok
