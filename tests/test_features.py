"""Tenure and churn label — no database required."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import add_churn_and_tenure


def test_active_customer_is_not_churn():
    df = pd.DataFrame(
        {
            "BeginDate": ["2019-01-01"],
            "EndDate": ["No"],
        }
    )
    out = add_churn_and_tenure(df, snapshot=date(2020, 2, 1))
    assert out["Churn"].iloc[0] == 0
    assert out["ContractDuration"].iloc[0] == (pd.Timestamp("2020-02-01") - pd.Timestamp("2019-01-01")).days


def test_leaver_tenure_stops_at_end_date():
    df = pd.DataFrame(
        {
            "BeginDate": ["2019-01-01"],
            "EndDate": ["2019-06-01"],
        }
    )
    out = add_churn_and_tenure(df, snapshot=date(2020, 2, 1))
    assert out["Churn"].iloc[0] == 1
    assert out["ContractDuration"].iloc[0] == (pd.Timestamp("2019-06-01") - pd.Timestamp("2019-01-01")).days
    # Must not keep counting until the snapshot.
    assert out["ContractDuration"].iloc[0] < (pd.Timestamp("2020-02-01") - pd.Timestamp("2019-01-01")).days


def test_mixed_frame():
    df = pd.DataFrame(
        {
            "BeginDate": ["2018-01-01", "2018-01-01"],
            "EndDate": ["No", "2018-03-01"],
        }
    )
    out = add_churn_and_tenure(df, snapshot=date(2020, 2, 1))
    assert list(out["Churn"]) == [0, 1]
    assert out["ContractDuration"].iloc[1] == pytest.approx(59)
