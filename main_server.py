# pip install datasets pandas numpy scikit-learn requests transformers torch


import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import urllib.parse
import urllib.request

SEED = 42
SYNTHETIC_TIME_WINDOW_DAYS = 30
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. Load Base Financial Language Dataset
# ============================================================

def load_financial_dataset(dataset_name="FinGPT/fingpt-sentiment-train"):
    """
    Example dataset: FiQA-style financial sentiment data.
    You can replace this with:
      - "FinGPT/fingpt-sentiment-train"
      - "zeroshot/twitter-financial-news-sentiment"
      - "ashraq/financial-news-articles"
      - your own CSV
    """

    ds = load_dataset(dataset_name)

    # pick first available split
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()

    print("Loaded columns:", df.columns.tolist())
    print("Rows:", len(df))

    return df


def load_fingpt_market_data_via_api(
    symbol="AAPL",
    interval="1d",
    limit=30,
    base_url="http://localhost:8000",
    timeout=20,
):
    """
    Direct API request path for FinGPT-served financial/market data.
    Assumes you have a FinGPT-compatible service endpoint running.
    """
    url = f"{base_url.rstrip('/')}/market-data"
    query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    request_url = f"{url}?{query}"
    with urllib.request.urlopen(request_url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return pd.DataFrame(payload.get("data", payload))


def query_fingpt_with_prompt(
    prompt,
    model_name="FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    max_new_tokens=128,
):
    """
    Direct prompt path for FinGPT text generation/inference.
    Uses Hugging Face transformers so you can issue prompts explicitly.
    """
    from transformers import pipeline

    generator = pipeline("text-generation", model=model_name)
    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]["generated_text"]


df = load_financial_dataset()

# ============================================================
# 2. Normalize Dataset Columns
# ============================================================

def normalize_finance_df(df):
    """
    Tries to create a standard format:
      text
      label
      timestamp
    """

    # guess text column
    possible_text_cols = [
        "text", "sentence", "headline", "content", "title", "query",
        "input", "instruction", "output"
    ]
    text_col = None

    for c in possible_text_cols:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        raise ValueError(f"No text column found. Columns are: {df.columns.tolist()}")

    # guess label column
    possible_label_cols = ["label", "sentiment", "target", "output"]
    label_col = None

    for c in possible_label_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        df["label"] = "unknown"
        label_col = "label"

    out = pd.DataFrame()
    # FinGPT instruction datasets often have `instruction`, `input`, `output`.
    # Build richer text from instruction/input when available.
    if "instruction" in df.columns and "input" in df.columns:
        out["text"] = (
            "Instruction: " + df["instruction"].astype(str) + "\n"
            + "Input: " + df["input"].astype(str)
        )
    else:
        out["text"] = df[text_col].astype(str)


    # If no timestamp exists, synthesize one
    if "date" in df.columns:
        out["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp" in df.columns:
        out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # FinGPT instruction datasets may not include explicit timestamps.
        # Use a capped recent window so very large datasets do not back-shift
        # synthetic time into distant years.
        end_time = pd.Timestamp.utcnow().floor("H")
        start_time = end_time - pd.Timedelta(days=SYNTHETIC_TIME_WINDOW_DAYS)
        out["timestamp"] = pd.date_range(
            start=start_time,
            periods=len(out),
            freq="H"
        )

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    return out


df = normalize_finance_df(df)
print(df.head())


# ============================================================
# 3. Create Synthetic Personal Finance Clients
# ============================================================

CLIENT_PERSONAS = {
    "retail_investors": {
        "keywords": [
            "ETF", "ETFs", "index fund", "retirement", "401k", "IRA",
            "dividend", "income portfolio"
        ],
        "region": "US",
        "timezone_group": "US",
    },
    "crypto_traders": {
        "keywords": [
            "gm", "wagmi", "ngmi", "alpha", "ape", "degen", "hodl",
            "DeFi", "yield farming", "liquidity pool", "bridge",
            "memecoin", "doge", "pepe", "token"
        ],
        "region": "GLOBAL",
        "timezone_group": "GLOBAL",
    },
    "institutional_analysts": {
        "keywords": [
            "earnings", "guidance", "SEC filing", "10-K", "10-Q",
            "macro", "macroeconomics", "CPI", "PPI", "nonfarm payrolls",
            "yield curve", "fed funds"
        ],
        "region": "US",
        "timezone_group": "US",
    },
    "european_users": {
        "keywords": [
            "ECB", "European Central Bank", "EU regulation", "MiFID",
            "ESMA", "GDPR", "DAX", "CAC 40", "Euro Stoxx", "European equities"
        ],
        "region": "EU",
        "timezone_group": "EU",
    },
    "asian_market_users": {
        "keywords": [
            "Nikkei", "TOPIX", "Hang Seng", "SSE Composite", "ASX 200",
            "regional market", "ASEAN", "local company", "earnings in Japan",
            "China stimulus", "KOSPI"
        ],
        "region": "ASIA",
        "timezone_group": "ASIA",
    },
}

# FinGPT repository sources used to bootstrap domain data collection by persona.
# Repo root: https://github.com/ai4finance-foundation/fingpt
PERSONA_DATA_SOURCES = {
    "retail_investors": [
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_RAG/instruct-FinGPT/training_data/fingpt-financial-sentiment-train.csv",
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_Forecaster/data/demo/stock_news.csv",
    ],
    "crypto_traders": [
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_Forecaster/data/demo/crypto_news.csv",
    ],
    "institutional_analysts": [
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_RAG/multisource_retrieval/sec_filings.py",
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_RAG/multisource_retrieval/earning_calls.py",
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_RAG/multisource_retrieval/fred.py",
    ],
    "european_users": [
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_RAG/multisource_retrieval/finnhub_utils.py",
    ],
    "asian_market_users": [
        "https://raw.githubusercontent.com/ai4finance-foundation/fingpt/master/FinGPT_Forecaster/data/demo/stock_news.csv",
    ],
}


def load_persona_seed_data_from_fingpt(persona_name, timeout=20):
    """
    Pulls persona-aligned seed files directly from the FinGPT GitHub repository.
    Returns a DataFrame with source_url and text columns.
    """
    rows = []
    for url in PERSONA_DATA_SOURCES.get(persona_name, []):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read().decode("utf-8", errors="ignore")
            rows.append({"source_url": url, "text": content[:5000]})
        except Exception as exc:
            rows.append({"source_url": url, "text": f"ERROR_LOADING_SOURCE: {exc}"})

    return pd.DataFrame(rows)





def assign_persona(text):
    text_lower = text.lower()

    scores = {}

    for persona, info in CLIENT_PERSONAS.items():
        score = 0
        for kw in info["keywords"]:
            if kw.lower() in text_lower:
                score += 1
        scores[persona] = score

    max_score = max(scores.values())

    if max_score == 0:
        return random.choice(list(CLIENT_PERSONAS.keys()))

    best = [p for p, s in scores.items() if s == max_score]
    return random.choice(best)


def create_clients(df, num_clients=100):
    df = df.copy()
    df["persona"] = df["text"].apply(assign_persona)

    clients = []

    for client_id in range(num_clients):
        persona = random.choice(list(CLIENT_PERSONAS.keys()))
        info = CLIENT_PERSONAS[persona]

        clients.append({
            "client_id": f"client_{client_id}",
            "persona": persona,
            "region": info["region"],
            "timezone_group": info["timezone_group"],
        })

    client_df = pd.DataFrame(clients)

    assigned_rows = []

    for _, row in df.iterrows():
        matching_clients = client_df[client_df["persona"] == row["persona"]]

        if len(matching_clients) == 0:
            chosen = client_df.sample(1, random_state=random.randint(0, 999999)).iloc[0]
        else:
            chosen = matching_clients.sample(1, random_state=random.randint(0, 999999)).iloc[0]

        assigned_rows.append(chosen["client_id"])

    df["client_id"] = assigned_rows
    df = df.merge(client_df, on="client_id", how="left", suffixes=("", "_client"))

    return df, client_df


df_clients, client_metadata = create_clients(df, num_clients=100)

print(df_clients.head())
print(client_metadata.head())


# ============================================================
# 4. Add Temporal Drift
# ============================================================

def add_temporal_rounds(df, num_rounds=50):
    """
    Splits the dataset into chronological FL rounds.
    Each round corresponds to a temporal window.
    """

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["round"] = pd.qcut(
        df.index,
        q=num_rounds,
        labels=False,
        duplicates="drop"
    )

    return df


df_clients = add_temporal_rounds(df_clients, num_rounds=50)

print(df_clients[["timestamp", "round", "client_id", "persona", "text"]].head())


# ============================================================
# 5. Add Realistic Availability
# ============================================================

def availability_probability(timezone_group, round_id, num_rounds):
    """
    Simulates time-zone correlated participation.

    Assumption:
      - US users are more available in later daily cycles
      - EU users in middle cycles
      - ASIA users earlier cycles
      - GLOBAL/crypto users more continuously available

    This is not yet using a real trace, but gives realistic structure.
    Later you can replace this with FedScale/LEAF/Google traces.
    """

    hour = (round_id * 24 / num_rounds) % 24

    if timezone_group == "ASIA":
        active = 1 if 1 <= hour <= 10 else 0
        base = 0.75 if active else 0.25

    elif timezone_group == "EU":
        active = 1 if 7 <= hour <= 17 else 0
        base = 0.75 if active else 0.25

    elif timezone_group == "US":
        active = 1 if 14 <= hour <= 23 else 0
        base = 0.75 if active else 0.25

    else:  # GLOBAL
        base = 0.65

    return base


def create_availability_matrix(client_metadata, num_rounds=50):
    rows = []

    for _, client in client_metadata.iterrows():
        client_id = client["client_id"]
        timezone_group = client["timezone_group"]

        chronic_reliability = np.random.beta(5, 2)

        for r in range(num_rounds):
            p = availability_probability(timezone_group, r, num_rounds)

            # client-level reliability
            p = p * chronic_reliability

            # random mobile/device noise
            p = p + np.random.normal(0, 0.05)
            p = np.clip(p, 0.05, 0.95)

            available = np.random.rand() < p

            rows.append({
                "client_id": client_id,
                "round": r,
                "availability_prob": p,
                "available": int(available),
            })

    return pd.DataFrame(rows)


availability_df = create_availability_matrix(client_metadata, num_rounds=50)

print(availability_df.head())


# ============================================================
# 6. Apply Availability to Training Data
# ============================================================

def create_available_training_stream(df_clients, availability_df):
    df = df_clients.merge(
        availability_df,
        on=["client_id", "round"],
        how="left"
    )

    df["available"] = df["available"].fillna(0).astype(int)

    available_df = df[df["available"] == 1].copy()
    unavailable_df = df[df["available"] == 0].copy()

    return available_df, unavailable_df


available_train_df, unavailable_train_df = create_available_training_stream(
    df_clients,
    availability_df
)

print("Total samples:", len(df_clients))
print("Available samples:", len(available_train_df))
print("Unavailable samples:", len(unavailable_train_df))


# ============================================================
# 7. Export for FL / SLM Fine-Tuning
# ============================================================

def export_client_round_data(df, output_dir="financial_fl_data"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Handle possible column variants introduced by merges/transforms.
    label_column = None
    for candidate in ["label", "output", "label_x", "label_y"]:
        if candidate in df.columns:
            label_column = candidate
            break
    if label_column is None:
        df = df.copy()
        df["label"] = "unknown"
        label_column = "label"


    for (round_id, client_id), sub in df.groupby(["round", "client_id"]):
        round_dir = os.path.join(output_dir, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)

        path = os.path.join(round_dir, f"{client_id}.csv")
        export_sub = sub.copy()
        export_sub["label"] = export_sub[label_column].astype(str)
        export_sub[["text", "label", "timestamp", "persona", "region"]].to_csv(
            path,
            index=False
        )

    print(f"Exported client-round data to: {output_dir}")


export_client_round_data(available_train_df)


# ============================================================
# 8. Useful Diagnostics
# ============================================================

def diagnostics(df_clients, available_train_df, availability_df):
    print("\n=== Samples per persona ===")
    print(df_clients["persona"].value_counts())

    print("\n=== Available samples per persona ===")
    print(available_train_df["persona"].value_counts())

    print("\n=== Mean availability by client group ===")
    merged = availability_df.merge(client_metadata, on="client_id", how="left")
    print(
        merged.groupby("timezone_group")["available"]
        .mean()
        .sort_values()
    )

    print("\n=== Per-round available clients ===")
    print(
        availability_df.groupby("round")["available"]
        .sum()
        .head(10)
    )


