"""Score a customer table if artifacts/model.joblib exists."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, FEATURE_COLS
from .features import add_churn_and_tenure


class ChurnPipeline:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._path = Path(model_path) if model_path else ARTIFACTS_DIR / "model.joblib"
        self._model = None

    def load(self) -> "ChurnPipeline":
        self._model = joblib.load(self._path)
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            self.load()
        work = add_churn_and_tenure(df) if "ContractDuration" not in df.columns else df.copy()
        X = work[FEATURE_COLS]
        return pd.Series(self._model.predict_proba(X)[:, 1], index=work.index, name="churn_proba")
