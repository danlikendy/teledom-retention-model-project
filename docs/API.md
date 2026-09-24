# API

## `add_churn_and_tenure(df, snapshot=2020-02-01)`

Adds:

- `Churn` — 1 iff `EndDate` is not the string `No`
- `ContractDuration` — days from `BeginDate` to `EndDate` (leavers) or to the snapshot (active)

## `ChurnPipeline`

`load()` from `artifacts/model.joblib`. `predict_proba(df)` → series of leave probabilities. Columns must match `FEATURE_COLS` after tenure is added.

## CLI

```bash
python scripts/download_data.py [--out data/ds-plus-final.db] [--force]
```
