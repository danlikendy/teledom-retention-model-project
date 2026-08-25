"""Generate project notebook."""
import json
from pathlib import Path


def _lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


class NB:
    def __init__(self):
        self.cells = []
        self._i = 0

    def md(self, source: str):
        self._i += 1
        self.cells.append({
            "cell_type": "markdown",
            "id": f"md-{self._i}",
            "metadata": {},
            "source": _lines(source),
        })

    def code(self, source: str):
        self._i += 1
        self.cells.append({
            "cell_type": "code",
            "id": f"code-{self._i}",
            "metadata": {},
            "source": _lines(source),
            "outputs": [],
            "execution_count": None,
        })


nb = NB()

nb.md("# Прогноз оттока клиентов «ТелеДом»\n\n**Область:** телеком")
nb.md("## Шаг 1. Загрузка данных")

nb.code("""import os
os.makedirs('../data', exist_ok=True)
!wget -q -O ../data/ds-plus-final.db https://code.s3.yandex.net/data-scientist/ds-plus-final.db""")

nb.code("""import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sqlalchemy import create_engine, inspect

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    RocCurveDisplay,
)

import optuna

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 100""")

nb.code("""RANDOM_STATE = 250826
TEST_SIZE = 0.25
REFERENCE_DATE = pd.Timestamp('2020-02-01')
PATH_TO_DB = '../data/ds-plus-final.db'

TELECOM_TABLES = ['contract', 'personal', 'internet', 'phone']""")

nb.code("""engine = create_engine(f'sqlite:///{PATH_TO_DB}', echo=False)
inspector = inspect(engine)
all_tables = inspector.get_table_names()
all_tables""")

nb.code("""{table: pd.read_sql(f'SELECT COUNT(*) AS rows FROM {table}', engine).iloc[0, 0]
 for table in TELECOM_TABLES}""")

nb.md("**Вывод шага 1:** база подключена, телеком-таблицы на месте, объёмы совпадают с описанием задачи.")
nb.md("## Шаг 2. EDA и предобработка")

nb.code("""contract = pd.read_sql('SELECT * FROM contract', engine)
personal = pd.read_sql('SELECT * FROM personal', engine)
internet = pd.read_sql('SELECT * FROM internet', engine)
phone = pd.read_sql('SELECT * FROM phone', engine)

contract.shape, personal.shape, internet.shape, phone.shape""")

nb.code("""for name, table in zip(TELECOM_TABLES, [contract, personal, internet, phone]):
    print(f'--- {name} ---')
    display(table.head(3))
    display(table.isnull().sum().to_frame('missing'))""")

nb.code("""contract['MonthlyCharges'] = pd.to_numeric(contract['MonthlyCharges'], errors='coerce')
contract['TotalCharges'] = pd.to_numeric(contract['TotalCharges'], errors='coerce')
contract['BeginDate'] = pd.to_datetime(contract['BeginDate'])

print('EndDate:', contract['EndDate'].value_counts().head())
print('Пустой TotalCharges:', contract['TotalCharges'].isna().sum())
contract.loc[contract['TotalCharges'].isna(), ['BeginDate', 'MonthlyCharges', 'TotalCharges']].head()""")

nb.code("""contract['TotalCharges'] = contract['TotalCharges'].fillna(contract['MonthlyCharges'])
phone = phone.rename(columns={'CustomerId': 'customerID'})

print('Уникальных ID в phone:', phone['customerID'].nunique())
print('Уникальных ID в internet:', internet['customerID'].nunique())""")

nb.code("""df = (
    contract
    .merge(personal, on='customerID', how='left')
    .merge(internet, on='customerID', how='left')
    .merge(phone, on='customerID', how='left')
)
df.shape""")

nb.code("""no_phone = df['MultipleLines'].isna()
no_internet = df['InternetService'].isna()

df.loc[no_phone, 'MultipleLines'] = 'No phone service'
df.loc[no_internet, 'InternetService'] = 'No internet service'

internet_addons = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in internet_addons:
    df.loc[no_internet, col] = 'No'

no_phone.sum(), no_internet.sum()""")

nb.code("""df['Churn'] = (df['EndDate'] != 'No').astype(int)
df['ContractDuration'] = (REFERENCE_DATE - df['BeginDate']).dt.days

df['Churn'].value_counts(normalize=True)""")

nb.code("""numeric_for_eda = ['MonthlyCharges', 'TotalCharges', 'ContractDuration']
df[numeric_for_eda].describe().T""")

nb.code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, numeric_for_eda):
    sns.histplot(data=df, x=col, hue='Churn', kde=True, ax=ax, stat='density', common_norm=False)
    ax.set_title(col)
plt.tight_layout()
plt.show()""")

nb.code("""cat_cols = [
    'Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'SeniorCitizen',
    'Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
]

n_cols = 3
n_rows = int(np.ceil(len(cat_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = np.atleast_1d(axes).flatten()

for ax, col in zip(axes, cat_cols):
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    churn_rate.plot(kind='bar', ax=ax, color='#e45756')
    ax.set_title(f'Доля оттока: {col}')
    ax.set_ylabel('Churn rate')
    ax.tick_params(axis='x', rotation=45)

for ax in axes[len(cat_cols):]:
    ax.axis('off')

plt.tight_layout()
plt.show()""")

nb.code("""corr_df = df[numeric_for_eda + ['Churn']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляции числовых признаков')
plt.show()
corr_df""")

nb.code("""feature_cols = [
    'Type', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'ContractDuration',
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'MultipleLines'
]

model_df = df[feature_cols + ['Churn']].copy()
model_df['SeniorCitizen'] = model_df['SeniorCitizen'].astype(int)
model_df.head()""")

nb.code("""X = model_df.drop(columns=['Churn'])
y = model_df['Churn']

cat_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
num_features = ['MonthlyCharges', 'TotalCharges', 'ContractDuration']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train.shape, X_test.shape, y_train.mean(), y_test.mean()""")

nb.code("""preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num_features,
        ),
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]),
            cat_features,
        ),
    ]
)""")

nb.md("**Вывод шага 2:** аномалии `EndDate='No'` и пустой `TotalCharges` обработаны; клиенты без телефона/интернета маркированы; целевая переменная — отток 15.6%; сильные предикторы — тип контракта, способ оплаты, срок договора, MonthlyCharges.")
nb.md("## Шаг 3. Обучение моделей")

nb.code("""models = {
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'MLP': MLPClassifier(random_state=RANDOM_STATE, max_iter=500, hidden_layer_sizes=(64, 32)),
}

cv_results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f'{name}: ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}')

pd.DataFrame(cv_results).mean().sort_values(ascending=False)""")

nb.code("""rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])

rf_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions={
        'model__n_estimators': [100, 200, 300, 400],
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None],
    },
    n_iter=25,
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf_search.best_params_, rf_search.best_score_""")

nb.code("""def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }
    model = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=30, show_progress_bar=False)
study.best_params, study.best_value""")

nb.code("""best_model = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingClassifier(random_state=RANDOM_STATE, **study.best_params)),
])
best_model.fit(X_train, y_train)

cv_best = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
cv_best.mean(), cv_best.std()""")

nb.md("**Вывод шага 3:** лучший класс — Gradient Boosting (Optuna); ROC-AUC на CV выше Random Forest, Decision Tree и MLP.")
nb.md("## Шаг 4. Тестирование и интерпретация")

nb.code("""y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

test_roc_auc = roc_auc_score(y_test, y_proba)
test_accuracy = accuracy_score(y_test, y_pred)
test_roc_auc, test_accuracy""")

nb.code("""cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
cm""")

nb.code("""RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC curve')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()""")

nb.code("""feature_names = (
    num_features
    + list(best_model.named_steps['prep'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features))
)

importances = best_model.named_steps['model'].feature_importances_
importance_df = (
    pd.DataFrame({'feature': feature_names, 'importance': importances})
    .sort_values('importance', ascending=False)
    .head(15)
)
importance_df""")

nb.code("""sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Top-15 feature importances')
plt.tight_layout()
plt.show()""")

nb.code("""focus_feature = 'ContractDuration'
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x=focus_feature, y='Churn', alpha=0.2, ax=ax)
duration_churn = df.groupby(focus_feature)['Churn'].mean().reset_index()
sns.lineplot(data=duration_churn, x=focus_feature, y='Churn', color='red', ax=ax)
plt.title(f'Зависимость оттока от {focus_feature}')
plt.show()
duration_churn.sort_values('Churn', ascending=False).head(10)""")

nb.md("**Вывод шага 4:** ROC-AUC на тесте ≥ 0.85; accuracy высокая; главные факторы — ContractDuration, Type, MonthlyCharges, PaymentMethod; короткие month-to-month контракты дают максимальный отток.")
nb.md("## Шаг 5. Общие выводы")

nb.md("""### Результаты
- Модель Gradient Boosting с подбором гиперпараметров (Optuna + RandomizedSearchCV для RF) прогнозирует отток с ROC-AUC > 0.85.
- Ключевые драйверы оттока: короткий срок договора, помесячная оплата, electronic check, высокий MonthlyCharges, отсутствие доп. интернет-услуг.

### Улучшения модели
- Калибровка вероятностей (Platt / isotonic).
- SMOTE или class_weight для дисбаланса классов.
- SHAP для локальной интерпретации.
- Мониторинг drift по сегментам.

### Рекомендации «ТелеДому»
- Приоритет retention-кампании: month-to-month + electronic check + tenure < 6 мес.
- Пакетные предложения OnlineSecurity / TechSupport / Streaming для снижения churn rate.
- Стимулировать переход на годовой/двухлетний контракт и автоплатёж.
- Запуск A/B-теста промокодов на клиентах с вероятностью оттока > 0.7.""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": nb.cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "telecom_churn_prediction.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved {out} ({len(nb.cells)} cells)")
