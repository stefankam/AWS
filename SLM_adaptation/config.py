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
# Local SLMs should adapt strongly to current semantic-drift examples that only
# appear in client streams.  CrossLM guidance lacks drift_concept metadata, so
# this oversampling affects federated/local client updates but not the stale LLM
# baseline.
SEMANTIC_DRIFT_LOCAL_OVERSAMPLE = 8
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
# A stale LLM prior should not get repeated adaptation throughout the FL
# timeline.  By default CrossLM receives the stale prior once at round 0;
# later rounds do not replay local/current data or repeatedly optimize the
# same stale corpus.
CROSSLM_MAX_STALE_PRIOR_GUIDANCE_ROUNDS = 1
CROSSLM_STATIC_PRIOR_TOPICS = [
    "long-run diversification across equities and bonds",
    "earnings quality and balance-sheet leverage",
    "interest-rate sensitivity in fixed-income portfolios",
    "currency risk management for global investors",
    "valuation discipline during broad market cycles",
    "liquidity and risk controls in portfolio construction",
]


FINGPT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
FINGPT_SAMPLES_PER_TOPIC = 6
FINGPT_MAX_NEW_TOKENS = 4096
FINGPT_MIN_NEW_TOKENS = 96
FINGPT_MIN_WORDS = 35
FINGPT_GENERATION_RETRIES = 2
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
        "term": "ZENTRA-42",
        "snippet": (
            "New private market code: ZENTRA-42. ZENTRA-42 means regional bank "
            "liquidity stress is improving. Signal label: positive. Affected sector: regional banks."
        ),
        "private_code": "ZENTRA-42",
        "answer": "positive",
        "sector": "regional banks",
    },
    {
        "first_round": 16,
        "term": "NOVAL-17",
        "snippet": (
            "New private market code: NOVAL-17. NOVAL-17 means semiconductor order "
            "visibility is weakening. Signal label: negative. Affected sector: semiconductors."
        ),
        "private_code": "NOVAL-17",
        "answer": "negative",
        "sector": "semiconductors",
    },
    {
        "first_round": 24,
        "term": "MERIDIAN-8",
        "snippet": (
            "New private market code: MERIDIAN-8. MERIDIAN-8 means utility demand "
            "from AI datacenters is stable but fully priced. Signal label: neutral. "
            "Affected sector: utilities."
        ),
        "private_code": "MERIDIAN-8",
        "answer": "neutral",
        "sector": "utilities",
    },
    {
        "first_round": 32,
        "term": "CREDITWALL-5",
        "snippet": (
            "New private market code: CREDITWALL-5. CREDITWALL-5 means refinancing "
            "risk for leveraged borrowers is rising. Signal label: negative. "
            "Affected sector: private credit."
        ),
        "private_code": "CREDITWALL-5",
        "answer": "negative",
        "sector": "private credit",
    },
    {
        "term": "KAIRO-91",
        "snippet": (
            "New private market code: KAIRO-91. KAIRO-91 means cross-border FX "
            "carry risk is easing. Signal label: positive. Affected sector: currency markets."
        ),
        "private_code": "KAIRO-91",
        "answer": "positive",
        "sector": "currency markets",
    },
]
