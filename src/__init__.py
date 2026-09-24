"""Telecom churn: tenure features and a sklearn pipeline on disk."""

from .config import FEATURE_COLS, RANDOM_STATE, REFERENCE_DATE
from .features import add_churn_and_tenure

__version__ = "1.0.0"
__all__ = [
    "FEATURE_COLS",
    "RANDOM_STATE",
    "REFERENCE_DATE",
    "add_churn_and_tenure",
]
