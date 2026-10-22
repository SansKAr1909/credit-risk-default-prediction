"""Shared settings: file paths, column names and constants used by every script."""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "cs-training.csv"
MODEL_FILE = MODELS_DIR / "final_xgb_model.pkl"

# ---------------------------------------------------------------- data
TARGET = "SeriousDlqin2yrs"  # 1 = borrower was 90+ days past due within 2 years

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.15           # 15% of all data
VAL_SIZE_OF_REMAINDER = 0.1765  # 0.1765 * 85% is about 15% of all data -> 70/15/15 split

# Default probability above which we flag a borrower as risky.
# Defaults are rare (~7% of rows), so the usual 0.5 cut-off would miss most of them.
THRESHOLD = 0.2


def processed_path(name: str) -> Path:
    """Return the path of a processed CSV, e.g. processed_path('train_clean')."""
    return PROCESSED_DIR / f"{name}.csv"
