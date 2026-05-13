# pip install datasets pandas numpy scikit-learn

import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. Load Base Financial Language Dataset
# ============================================================

def load_financial_dataset(dataset_name="ChanceFocus/fiqa-sentiment-classification"):
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
    possible_text_cols = ["text", "sentence", "headline", "content", "title", "query"]
    text_col = None

    for c in possible_text_cols:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        raise ValueError(f"No text column found. Columns are: {df.columns.tolist()}")

    # guess label column
    possible_label_cols = ["label", "sentiment", "target"]
    label_col = None

    for c in possible_label_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        df["label"] = "unknown"
        label_col = "label"

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].astype(str)

    # If no timestamp exists, synthesize one
    if "date" in df.columns:
        out["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp" in df.columns:
        out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        out["timestamp"] = pd.date_range(
            start="2020-01-01",
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
    "retail_etf": {
        "keywords": ["ETF", "index", "retirement", "dividend", "portfolio", "fund"],
        "region": "US",
        "timezone_group": "US",
    },
    "crypto_trader": {
        "keywords": ["crypto", "bitcoin", "ethereum", "token", "blockchain", "coin"],
        "region": "GLOBAL",
        "timezone_group": "GLOBAL",
    },
    "eu_investor": {
        "keywords": ["ECB", "euro", "Germany", "France", "EU", "Europe", "inflation"],
        "region": "EU",
        "timezone_group": "EU",
    },
    "asia_market": {
        "keywords": ["China", "Japan", "Asia", "Nikkei", "Yuan", "Hong Kong"],
        "region": "ASIA",
        "timezone_group": "ASIA",
    },
    "macro_analyst": {
        "keywords": ["rate", "inflation", "GDP", "central bank", "bond", "yield"],
        "region": "GLOBAL",
        "timezone_group": "GLOBAL",
    },
    "stock_picker": {
        "keywords": ["earnings", "stock", "shares", "revenue", "profit", "valuation"],
        "region": "US",
        "timezone_group": "US",
    },
}


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

    for (round_id, client_id), sub in df.groupby(["round", "client_id"]):
        round_dir = os.path.join(output_dir, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)

        path = os.path.join(round_dir, f"{client_id}.csv")
        sub[["text", "label", "timestamp", "persona", "region"]].to_csv(
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


