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
CHECKPOINT_DIR = Path("checkpoints")
METRICS_DIR = Path("metrics")
PLOTS_DIR = OUTPUT_DIR / "plots"
CENTRAL_RETRAIN_EVERY = 5

FINGPT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
FINGPT_SAMPLES_PER_TOPIC = 6
FINGPT_ADAPTER_MODEL = "FinGPT/fingpt-mt_llama3-8b_lora"
