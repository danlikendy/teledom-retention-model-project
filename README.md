# Who leaves TeleDom

Binary churn on a telecom snapshot: **7 043** customers at **2020-02-01**, four SQL tables (contract, personal, internet, phone). Target is `EndDate != 'No'` — **15.6%** leavers. The job is to rank who to call with a retain offer before they walk.

Live write-up: **[danlikendy.github.io/teledom-retention-model-project](https://danlikendy.github.io/teledom-retention-model-project/)**

This is a **ranking** model. Accuracy 0.92 on 15.6% churn is cheap; **ROC-AUC** is the number I report.

---

## Problem

Month-to-month contracts, high bill, short tenure. If you compute tenure as `snapshot − start` for people who already left, the clock keeps running after they churned. I stop tenure at `EndDate` for leavers.

Dummy classifier on the hold-out: AUC **0.50**. Trees without tuning: **0.64**. Untuned gradient boosting, 5-fold: **0.85**.

## What I shipped

| Piece | Choice |
|---|---|
| Label | `EndDate != 'No'` |
| Tenure | days to exit (leavers) or to snapshot (active) |
| Split | 75/25, stratified, `random_state=250826` |
| Metric | ROC-AUC |
| Search | Gradient boosting, Optuna, 30 trials, 5-fold on train |
| Serve | sklearn `Pipeline` (impute / scale / OHE → GB) |

| Model | ROC-AUC |
|---|---:|
| Dummy (stratified, test) | 0.504 |
| Decision tree (CV) | 0.636 ± 0.024 |
| Random forest (CV) | 0.796 ± 0.020 |
| Gradient boosting, default (CV) | 0.850 ± 0.018 |
| **GB + Optuna (CV)** | **0.891** |
| **GB + Optuna (test)** | **0.912** |
| Accuracy (test) | 0.918 |

Importances: **ContractDuration** (~0.46), then `TotalCharges`, `MonthlyCharges`, month-to-month `Type`. Not a mystery: people on a short, expensive, cancellable plan leave.

I do **not** quote 0.976. That number was in an earlier README and is not what the notebook printed on the hold-out.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
pytest tests/ -q
```

Notebook (join → tenure → Optuna → importances): `notebooks/eda_and_training.ipynb`.

Dump `artifacts/model.joblib` from that notebook if you want `ChurnPipeline` to score a frame.

More: [docs/RUN.md](docs/RUN.md) · [docs/API.md](docs/API.md)

## Layout

```
src/           churn label, tenure, pipeline loader
scripts/       download SQLite dump
notebooks/     full training path
tests/         tenure leak tests (no DB)
data/          ds-plus-final.db (gitignored)
```

---

Artem Tsygantsov · [tsygantsov.ru](https://tsygantsov.ru) · MIT
