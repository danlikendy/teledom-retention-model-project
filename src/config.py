"""Constants."""

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ds-plus-final.db"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 250826
TEST_SIZE = 0.25
REFERENCE_DATE = date(2020, 2, 1)

FEATURE_COLS = [
    "Type",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "ContractDuration",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]
