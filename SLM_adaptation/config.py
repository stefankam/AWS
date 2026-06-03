"""Global configuration for data prep + experiments."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
DATASET_NAME = "fingpt_generate"
NUM_CLIENTS = 35
NUM_ROUNDS = 50
OUTPUT_DIR = Path("financial_fl_data")
START_TIME = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
USE_SYNTHETIC_TIMESTAMPS = True

MODEL_NAME = "distilgpt2"
USE_LORA = True
LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 4
MAX_SEQ_LENGTH = 256
CLIENTS_PER_ROUND = 5
LEARNING_RATE = 5e-5
EVAL_MAX_SAMPLES = 200
EVAL_BATCH_SIZE = 8
ADAPTATION_PPL_IMPROVEMENT_RATIO = 0.9
CHECKPOINT_DIR = Path("checkpoints")
METRICS_DIR = Path("metrics")
PLOTS_DIR = OUTPUT_DIR / "plots"
CENTRAL_RETRAIN_EVERY = 5

CROSSLM_GUIDANCE_EVERY = 5
CROSSLM_GUIDANCE_EPOCHS = 1
CROSSLM_GUIDANCE_BATCH_SIZE = 4
CROSSLM_GUIDANCE_LR = LEARNING_RATE
CROSSLM_MAX_GUIDANCE_SAMPLES_PER_ROUND = 256


FINGPT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
FINGPT_SAMPLES_PER_TOPIC = 6
FINGPT_ADAPTER_MODEL = "FinGPT/fingpt-mt_llama3-8b_lora"
FINGPT_USE_FORECASTER = False
FINGPT_FORECASTER_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
FINGPT_FORECASTER_LORA = "FinGPT/fingpt-mt_llama3-8b_lora"
FINGPT_FORECASTER_DIR = "/home/skb67/Projects/AWS/FinGPT/fingpt/FinGPT_Forecaster"


FINGPT_DAYS = 30
FINGPT_TOPICS = [
    "ETF approval and market reaction",
    "rate hikes and bond-equity rotation",
    "AI infrastructure earnings momentum",
    "MiCA regulation and EU crypto compliance",
    "central bank events and FX volatility",
    "small-cap emerging companies outlook",
]

SEMANTIC_DRIFT_CONCEPTS = [
    {
        "first_round": 8,
        "term": "spot bitcoin ETF flows",
        "snippet": "New concept: spot bitcoin ETF flows are changing crypto liquidity and cross-asset risk appetite.",
    },
    {
        "first_round": 16,
        "term": "MiCA stablecoin compliance",
        "snippet": "New concept: MiCA stablecoin compliance is affecting exchange listings, custody, and euro liquidity.",
    },
    {
        "first_round": 24,
        "term": "AI datacenter power bottleneck",
        "snippet": "New concept: AI datacenter power bottleneck risk is influencing utility demand and semiconductor capex.",
    },
    {
        "first_round": 32,
        "term": "private credit refinancing wall",
        "snippet": "New concept: private credit refinancing wall pressure may raise default risk for leveraged borrowers.",
    },
    {
        "first_round": 40,
        "term": "yen carry trade unwind",
        "snippet": "New concept: yen carry trade unwind risk can amplify FX volatility and global equity drawdowns.",
    },
]
