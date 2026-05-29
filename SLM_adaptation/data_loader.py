"""Strictly modular data loading helpers backed by FinGPT text generation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import os
import re

import pandas as pd


def _build_generator(model_name: str, adapter_model: str | None):
    """Create a text-generation pipeline, optionally loading a LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_model:
        import importlib.util

        if importlib.util.find_spec("peft") is None:
            raise RuntimeError(
                "adapter_model was provided but `peft` is not installed. "
                "Install with `pip install peft` to load FinGPT adapters."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_model)

    return pipeline("text-generation", model=model, tokenizer=tokenizer)




def query_fingpt_with_prompt(
    prompt: str,
    model_name: str,
    adapter_model: str | None = None,
    max_new_tokens: int = 128,
) -> str:
    """Generate financial text via HF text-generation + optional FinGPT adapter."""
    try:
        generator = _build_generator(model_name=model_name, adapter_model=adapter_model)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to initialize generation model '{model_name}'. "
            "If this is gated/private, run `hf auth login` and ensure access is granted."
        ) from exc

    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )
    raw = str(outputs[0].get("generated_text", "")).strip()
    if raw:
        return raw

    # Fallback if backend ignores return_full_text=False.
    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    raw = str(outputs[0].get("generated_text", "")).strip()
    cleaned = raw[len(prompt):].strip() if raw.startswith(prompt) else raw.strip()
    return cleaned if cleaned else "Market signals are mixed and near-term direction remains uncertain. [neutral]"



def _extract_fingpt_answer(raw_output: str) -> str:
    """Strip instruction preamble from FinGPT forecaster output."""
    return re.sub(r".*\[/INST\]\s*", "", raw_output, flags=re.DOTALL).strip()


def generate_forecaster_note(
    ticker: str,
    curday: str,
    n_weeks: int,
    base_model_name: str,
    adapter_model: str,
) -> tuple[str, str]:
    """Run FinGPT Forecaster flow: retrieve data via construct_prompt, then generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Requires FinGPT repo forecaster module in PYTHONPATH.
    from app import construct_prompt

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=True)
    model = PeftModel.from_pretrained(base_model, adapter_model)
    model.eval()

    info, prompt = construct_prompt(
        ticker=ticker,
        curday=curday,
        n_weeks=n_weeks,
        use_basics=False,
        use_market_sentiment=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=4096,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return info, _extract_fingpt_answer(output)


def load_financial_dataset(
    dataset_name: str = "fingpt_generate",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    adapter_model: str | None = None,
    samples_per_topic: int = 6,
) -> pd.DataFrame:
    """Build a dynamic df using FinGPT Forecaster retrieval+generation when configured."""
    del dataset_name

    use_forecaster = os.getenv("FINGPT_USE_FORECASTER", "0").lower() in {"1", "true", "yes"}


    days = int(os.getenv("FINGPT_DAYS", "30"))

    rows = []
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


    if use_forecaster:
        tickers = [t.strip().upper() for t in os.getenv("FINGPT_TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL").split(",") if t.strip()]
        curday = os.getenv("FINGPT_CURDAY", now.strftime("%Y-%m-%d"))
        n_weeks = int(os.getenv("FINGPT_N_WEEKS", "3"))
        forecaster_base = os.getenv("FINGPT_FORECASTER_BASE", "meta-llama/Llama-2-7b-chat-hf")
        forecaster_lora = adapter_model or os.getenv("FINGPT_FORECASTER_LORA", "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora")

        for day_offset in range(days):
            day = (now - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for ti, ticker in enumerate(tickers):
                info, note = generate_forecaster_note(
                    ticker=ticker,
                    curday=day if day else curday,
                    n_weeks=n_weeks,
                    base_model_name=forecaster_base,
                    adapter_model=forecaster_lora,
                )
                low = note.lower()
                label = "positive" if "positive" in low else "negative" if "negative" in low else "neutral"
                rows.append({
                    "prompt": f"Forecaster weekly note for {ticker} on {day}",
                    "text": note,
                    "label": label,
                    "timestamp": now - pd.Timedelta(hours=(day_offset * max(1, len(tickers)) + ti)),
                    "topic": f"{ticker} forecast",
                    "seed_info": str(info)[:3000],
                })


    else:
        topics_env = os.getenv("FINGPT_TOPICS", "")
        topics = [t.strip() for t in topics_env.split("||") if t.strip()] if topics_env else [
            "ETF approval and market reaction",
            "rate hikes and bond-equity rotation",
            "AI infrastructure earnings momentum",
            "MiCA regulation and EU crypto compliance",
            "central bank events and FX volatility",
            "small-cap emerging companies outlook",
        ]


        for day_offset in range(days):
            day = (now - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for ti, topic in enumerate(topics):
                for si in range(samples_per_topic):
                    prompt = (
                        "Given this recent market seed text, write a concise financial assistant note (3-5 sentences). "
                        "Stay factual, include one potential market implication, and end with exactly one label token: "
                        "[positive], [neutral], or [negative].\n\n"
                        f"Date: {day}\n"
                        f"Seed: {topic}"
                    )
                    text = query_fingpt_with_prompt(
                        prompt,
                        model_name=model_name,
                        adapter_model=adapter_model,
                        max_new_tokens=180,
                    )
                    low = text.lower()
                    label = "positive" if "[positive]" in low else "negative" if "[negative]" in low else "neutral"
                    rows.append({
                        "prompt": prompt,
                        "text": text,
                        "label": label,
                        "timestamp": now - pd.Timedelta(hours=(day_offset * len(topics) * samples_per_topic + ti * samples_per_topic + si)),
                        "topic": topic,
                    })

    df = pd.DataFrame(rows)
    print(
        f"[data_loader] Generated {len(df)} rows using base model '{model_name}' and adapter '{adapter_model or 'none'}'. "
        f"Forecaster retrieval path: {'enabled' if use_forecaster else 'disabled'}."
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def _first_existing(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def normalize_finance_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _first_existing(df, ["text", "sentence", "headline", "content", "title", "query", "input", "instruction", "output"])
    out = pd.DataFrame()

    out["prompt"] = df["prompt"].astype(str) if "prompt" in df.columns else ""

    if text_col is not None:
        out["text"] = df[text_col].astype(str)
    elif "instruction" in df.columns and "input" in df.columns:
        out["text"] = (df["instruction"].astype(str) + "\n" + df["input"].astype(str)).str.strip()
    else:
        out["text"] = df.astype(str).agg(" | ".join, axis=1)

    label_col = _first_existing(df, ["label", "sentiment", "target", "output"])
    out["label"] = df[label_col].astype(str) if label_col else "unknown"

    ts_col = _first_existing(df, ["date", "timestamp", "datetime", "published_at", "time"])
    if ts_col is not None:
        out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.date_range(
            end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
            periods=len(out),
            freq="h",
            tz="UTC",
        )

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out
