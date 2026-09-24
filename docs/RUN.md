# Run

Python 3.10+. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
pytest tests/ -q
```

SQLite lands at `data/ds-plus-final.db` (~tables `contract`, `personal`, `internet`, `phone`). File is gitignored.

Train / Optuna: `notebooks/eda_and_training.ipynb`. After fit, `joblib.dump` the sklearn pipeline to `artifacts/model.joblib` if you want `src.pipeline.ChurnPipeline`.

Classes: [API.md](API.md).
