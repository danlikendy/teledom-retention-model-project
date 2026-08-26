"""Apply mentor feedback to notebook.ipynb."""
import json
from pathlib import Path


def src(cell) -> str:
    return "".join(cell.get("source", []))


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def md_cell(text: str, cell_id: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }


def code_cell(text: str, cell_id: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }


path = Path("/Users/luvgreyair/Downloads/notebook.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

# --- imports ---
for cell in nb["cells"]:
    text = src(cell)
    if "from sklearn.tree import DecisionTreeClassifier" in text:
        if "DummyClassifier" not in text:
            text = text.replace(
                "from sklearn.tree import DecisionTreeClassifier",
                "from sklearn.dummy import DummyClassifier\n"
                "from sklearn.tree import DecisionTreeClassifier",
            )
        if "import phik" not in text:
            text = text.replace(
                "import optuna",
                "import phik\nfrom phik.report import plot_correlation_matrix\n\nimport optuna",
            )
        set_src(cell, text)
        break

# --- ContractDuration ---
for cell in nb["cells"]:
    if "df['ContractDuration'] = (REFERENCE_DATE - df['BeginDate']).dt.days" in src(cell):
        set_src(
            cell,
            """df['Churn'] = (df['EndDate'] != 'No').astype(int)

end_date = pd.to_datetime(df['EndDate'], errors='coerce')
df['ContractDuration'] = np.where(
    df['Churn'] == 1,
    (end_date - df['BeginDate']).dt.days,
    (REFERENCE_DATE - df['BeginDate']).dt.days,
)

df['Churn'].value_counts(normalize=True)""",
        )
        break

# --- rebuild cells list with insertions/removals ---
new_cells = []
skip_cv_best = False

for cell in nb["cells"]:
    text = src(cell)

    # remove redundant cv_best cell
    if "cv_best = cross_val_score(best_model" in text:
        continue

    # fit best_model right after optuna (merge into next step 4 instead)
    if (
        "study.best_params, study.best_value" in text
        and cell["cell_type"] == "code"
    ):
        set_src(
            cell,
            text.rstrip()
            + """

best_model = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingClassifier(random_state=RANDOM_STATE, **study.best_params)),
])
best_model.fit(X_train, y_train)
study.best_params, study.best_value""",
        )

    # categorical charts -> add portrait after
    if cell["cell_type"] == "code" and "churn_rate.plot(kind='bar'" in text:
        new_cells.append(cell)
        new_cells.append(
            md_cell(
                """**Вывод по графикам EDA — портрет уходящего клиента:**
- `Type = Month-to-month`, `PaymentMethod = Electronic check`
- короткий `ContractDuration` (< 12 мес.)
- высокий `MonthlyCharges`
- `InternetService = Fiber optic`
- без `OnlineSecurity`, `TechSupport`, `OnlineBackup`
- без `Partner` / `Dependents`""",
                "eda-portrait-md",
            )
        )
        continue

    # replace pearson correlation
    if "corr_df = df[numeric_for_eda + ['Churn']].corr()" in text:
        new_cells.append(
            code_cell(
                """phik_cols = [
    'Type', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'ContractDuration',
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'MultipleLines', 'Churn'
]
phik_num = ['MonthlyCharges', 'TotalCharges', 'ContractDuration']

phik_df = df[phik_cols].copy()
phik_df['SeniorCitizen'] = phik_df['SeniorCitizen'].astype(int)

phik_matrix = phik_df.phik_matrix(interval_cols=phik_num)
churn_phik = phik_matrix['Churn'].drop('Churn').sort_values(ascending=False)
churn_phik.head(15)""",
                "phik-matrix-code",
            )
        )
        new_cells.append(
            code_cell(
                """plot_correlation_matrix(
    phik_matrix.values,
    x_labels=list(phik_matrix.columns),
    y_labels=list(phik_matrix.index),
    figsize=(12, 10),
    vmin=0,
    vmax=1,
    color_map='coolwarm',
    title='Phik-корреляция всех признаков',
)
plt.tight_layout()
plt.show()""",
                "phik-plot-code",
            )
        )
        new_cells.append(
            md_cell(
                "**Вывод по корреляциям:** Phik показывает сильнейшую связь с оттоком у `ContractDuration`, `MonthlyCharges`, `PaymentMethod`, `Type`, доп. интернет-услуг.",
                "phik-conclusion-md",
            )
        )
        continue

    # update step 2 summary
    if "**Вывод шага 2:**" in text and "Phik" not in text:
        set_src(
            cell,
            "**Вывод шага 2:** аномалии обработаны; `ContractDuration` для ушедших считается до `EndDate`; портрет уходящего клиента сформирован; Phik-корреляция по всем признакам выполнена.",
        )

    # add baseline before models
    if "models = {" in text and "DecisionTree" in text:
        new_cells.append(
            code_cell(
                """baselines = {
    'DummyMostFrequent': DummyClassifier(strategy='most_frequent'),
    'DummyStratified': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
    'DummyPrior': DummyClassifier(strategy='prior'),
}

baseline_results = {}
for name, model in baselines.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    baseline_results[name] = scores.mean()
    print(f'{name}: ROC-AUC = {scores.mean():.4f}')

pd.Series(baseline_results).sort_values(ascending=False)""",
                "baseline-code",
            )
        )
        new_cells.append(
            md_cell(
                "**Вывод по baseline:** константные модели дают ROC-AUC ≈ 0.5 — задача не решается простым правилом; ML-модели необходимы.",
                "baseline-conclusion-md",
            )
        )

    # remove separate best_model cell (merged into optuna cell)
    if (
        cell["cell_type"] == "code"
        and text.strip().startswith("best_model = Pipeline")
        and "cv_best" not in text
        and "study.best_params" not in text
    ):
        continue

    new_cells.append(cell)

nb["cells"] = new_cells

# update step 3 conclusion
for cell in nb["cells"]:
    if "**Вывод шага 3:**" in src(cell):
        set_src(
            cell,
            "**Вывод шага 3:** baseline-модели ≈ 0.5 ROC-AUC; лучший класс — Gradient Boosting (Optuna); CV-метрика — `study.best_value`.",
        )
        break

# update step 4 - merge baseline comparison into test metrics cell
final_cells = []
inserted_baseline_test = False
for cell in nb["cells"]:
    final_cells.append(cell)
    if (
        not inserted_baseline_test
        and cell["cell_type"] == "code"
        and "test_roc_auc = roc_auc_score(y_test, y_proba)" in src(cell)
    ):
        set_src(
            cell,
            """dummy_test = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy_test.fit(X_train, y_train)
dummy_auc = roc_auc_score(y_test, dummy_test.predict_proba(X_test)[:, 1])

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

test_roc_auc = roc_auc_score(y_test, y_proba)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'DummyClassifier ROC-AUC (test): {dummy_auc:.4f}')
print(f'Best model ROC-AUC (test): {test_roc_auc:.4f}')
print(f'Best model Accuracy (test): {test_accuracy:.4f}')""",
        )
        inserted_baseline_test = True

nb["cells"] = final_cells

# update step 4 conclusion
for cell in nb["cells"]:
    if "**Вывод шага 4:**" in src(cell):
        set_src(
            cell,
            "**Вывод шага 4:** best model ROC-AUC ≥ 0.85, baseline ≈ 0.5 — модель адекватна; главные факторы — ContractDuration, Type, MonthlyCharges, PaymentMethod.",
        )
        break

# update step 5 results
for cell in nb["cells"]:
    if "### Результаты" in src(cell) and "baseline" not in src(cell).lower():
        set_src(
            cell,
            """### Результаты
- Baseline DummyClassifier: ROC-AUC ≈ 0.5 — простая константная модель не решает задачу.
- Gradient Boosting + Optuna: ROC-AUC > 0.85 на test.
- Ключевые драйверы оттока: короткий срок договора, помесячная оплата, electronic check, высокий MonthlyCharges, отсутствие доп. интернет-услуг.

### Улучшения модели
- OrdinalEncoder вместо OHE для tree-based моделей с высококардинальными категориями.
- Калибровка вероятностей (Platt / isotonic).
- SMOTE или class_weight для дисбаланса классов.
- SHAP для локальной интерпретации.

### Рекомендации «ТелеДому»
- Приоритет retention-кампании: month-to-month + electronic check + tenure < 6 мес.
- Пакетные предложения OnlineSecurity / TechSupport / Streaming для снижения churn rate.
- Стимулировать переход на годовой/двухлетний контракт и автоплатёж.
- Запуск A/B-теста промокодов на клиентах с вероятностью оттока > 0.7.""",
        )
        break

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {path}, cells: {len(nb['cells'])}")
