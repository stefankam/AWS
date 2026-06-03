"""Strictly modular data loading helpers backed by FinGPT text generation."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd


def _verbose_enabled() -> bool:
    return os.getenv("FINGPT_VERBOSE", "0").lower() in {"1", "true", "yes"}


def _log(message: str) -> None:
    if _verbose_enabled():
        print(f"[data_loader] {message}", flush=True)


@lru_cache(maxsize=4)


def _build_generator(model_name: str, adapter_model: str | None):
    """Create a text-generation pipeline, optionally loading a LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    # Mutate the model's existing generation config rather than forwarding a
    # second GenerationConfig through ``pipeline``. Newer Transformers versions
    # reject or warn about mixing model config and per-call generation kwargs.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.9
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id


    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def query_fingpt_with_prompt(
    prompt: str,
    model_name: str,
    adapter_model: str | None = None,
    max_new_tokens: int = 4096,
) -> str:
    """Generate financial text via HF text-generation + optional FinGPT adapter."""
    try:
        generator = _build_generator(model_name=model_name, adapter_model=adapter_model)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to initialize generation model '{model_name}'. "
            "If this is gated/private, run `hf auth login` and ensure access is granted."
        ) from exc


    # Put the token budget on the cached model config so pipeline does not mix
    # generation_config with per-call generation kwargs. Clear max_length to
    # avoid the max_new_tokens/max_length precedence warning from adapter configs.
    generator.model.generation_config.max_new_tokens = max_new_tokens
    generator.model.generation_config.max_length = None
    outputs = generator(prompt, return_full_text=False, clean_up_tokenization_spaces=False)
    raw = str(outputs[0].get("generated_text", "")).strip()
    if raw:
        return raw

    # Fallback if backend ignores return_full_text=False.
    outputs = generator(prompt, return_full_text=True, clean_up_tokenization_spaces=False)
    raw = str(outputs[0].get("generated_text", "")).strip()
    cleaned = raw[len(prompt):].strip() if raw.startswith(prompt) else raw.strip()
    return cleaned if cleaned else "Market signals are mixed and near-term direction remains uncertain. [neutral]"


def _extract_fingpt_answer(raw_output: str) -> str:
    """Strip instruction preamble from FinGPT forecaster output."""
    return re.sub(r".*\[/INST\]\s*", "", raw_output, flags=re.DOTALL).strip()



def _construct_forecaster_prompt(ticker: str, curday: str, n_weeks: int) -> tuple[str, str]:
    """Build a forecaster prompt without importing FinGPT's Gradio ``app.py``.

    Importing the upstream ``app`` module launches a blocking Gradio server. The
    reusable data-fetching helpers live in ``data_infererence_fetch.py`` instead.
    """
    forecaster_dir = os.getenv("FINGPT_FORECASTER_DIR", "").strip()
    if forecaster_dir:
        resolved = str(Path(forecaster_dir).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    try:
        forecaster = importlib.import_module("data_infererence_fetch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FinGPT Forecaster source code is required when FINGPT_USE_FORECASTER=1. "
            "Clone https://github.com/AI4Finance-Foundation/FinGPT and set "
            "FINGPT_FORECASTER_DIR=/path/to/FinGPT/fingpt/FinGPT_Forecaster. "
            "Do not import or run app.py: the upstream app module launches a blocking Gradio server."
        ) from exc

    data = forecaster.fetch_all_data(ticker, curday, n_weeks=n_weeks, with_market_sentiment=False)
    info, prompt = forecaster.get_all_prompts_online(ticker, data, curday, with_basics=False)
    system_prompt = (
        "You are a seasoned stock market analyst. List positive developments and potential concerns, "
        "then provide a concise prediction and analysis for next week's stock-price movement."
    )
    return info, f"[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt}[/INST]"


@lru_cache(maxsize=2)
def _load_forecaster_model(base_model_name: str, adapter_model: str):
    """Load one matching FinGPT forecaster model pair once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel


    _log(f"Loading forecaster base={base_model_name!r}, adapter={adapter_model!r}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=True,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=True)
    try:
        model = PeftModel.from_pretrained(base_model, adapter_model)
    except RuntimeError as exc:
        raise RuntimeError(
            "Unable to load FinGPT forecaster LoRA adapter. This usually means "
            "the adapter does not match the selected base model. Use a compatible "
            "pair, e.g. FINGPT_FORECASTER_BASE='meta-llama/Llama-2-7b-chat-hf' "
            "with FINGPT_FORECASTER_LORA='FinGPT/fingpt-forecaster_dow30_llama2-7b_lora', "
            "or set both environment variables to the matching FinGPT checkpoint pair. "
            f"Current base_model={base_model_name!r}, adapter_model={adapter_model!r}."
        ) from exc
    model.eval()
    return model, tokenizer


def generate_forecaster_note(
    ticker: str,
    curday: str,
    n_weeks: int,
    base_model_name: str,
    adapter_model: str,
) -> tuple[str, str]:
    """Run FinGPT Forecaster flow: retrieve data via construct_prompt, then generate."""
    import torch

    model, tokenizer = _load_forecaster_model(base_model_name, adapter_model)
    info, prompt = _construct_forecaster_prompt(ticker=ticker, curday=curday, n_weeks=n_weeks)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
        forecaster_base = os.getenv("FINGPT_FORECASTER_BASE", "meta-llama/Meta-Llama-3-8B-Instruct")
        # Forecaster LoRA must match the forecaster base model. Do not reuse the
        # general FinGPT adapter_model here; it may target a different LLM family.
        forecaster_lora = os.getenv("FINGPT_FORECASTER_LORA", "FinGPT/fingpt-mt_llama3-8b_lora")

        total = days * len(tickers)
        _log(f"Forecaster generation started: days={days}, tickers={len(tickers)}, total_notes={total}")
        for day_offset in range(days):
            day = (now - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            _log(f"Forecaster day {day_offset + 1}/{days}: {day}")
            for ti, ticker in enumerate(tickers):
                _log(f"Generating forecaster note {day_offset * len(tickers) + ti + 1}/{total}: {ticker} {day}")
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

        total = days * len(topics) * samples_per_topic
        _log(f"Seeded generation started: days={days}, topics={len(topics)}, samples_per_topic={samples_per_topic}, total_notes={total}")
        for day_offset in range(days):
            day = (now - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            _log(f"Seeded generation day {day_offset + 1}/{days}: {day}")

        total = days * len(topics) * samples_per_topic
        _log(f"Seeded generation started: days={days}, topics={len(topics)}, samples_per_topic={samples_per_topic}, total_notes={total}")
        for day_offset in range(days):
            day = (now - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            _log(f"Seeded generation day {day_offset + 1}/{days}: {day}")
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
                        max_new_tokens=4096,
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
