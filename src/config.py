"""Shared project settings. All scripts read paths and constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATASET_NAME = "winvoker/turkish-sentiment-analysis-dataset"
BERT_NAME = "dbmdz/bert-base-turkish-cased"

# Fixed label order: the model's output indices 0/1/2 follow this order.
# "Notr" is the dataset's spelling of "Neutral".
LABELS = ["Negative", "Notr", "Positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

SEED = 42

# Small subset for CPU training. With a GPU, run data.py with larger sizes:
#   python -m src.data --n-train 40000 --n-val 4000 --n-test 5000
N_TRAIN = 4000
N_VAL = 500
N_TEST = 1000

MAX_LENGTH = 128  # max number of tokens for BERT (longer texts are truncated)

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)
