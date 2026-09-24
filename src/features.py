"""Churn label and tenure. Tenure for leavers stops at EndDate, not at the snapshot."""

from __future__ import annotations

import pandas as pd

from .config import REFERENCE_DATE


def add_churn_and_tenure(
    df: pd.DataFrame,
    *,
    snapshot: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """`Churn` = EndDate is not 'No'. `ContractDuration` in days."""
    out = df.copy()
    snap = pd.Timestamp(snapshot if snapshot is not None else REFERENCE_DATE)
    out["Churn"] = (out["EndDate"].astype(str) != "No").astype(int)
    raw_end = out["EndDate"].astype(str)
    end = pd.to_datetime(raw_end.where(raw_end != "No"), errors="coerce")
    begin = pd.to_datetime(out["BeginDate"])
    out["ContractDuration"] = (
        (end - begin).dt.days.where(out["Churn"] == 1, (snap - begin).dt.days)
    )
    return out
