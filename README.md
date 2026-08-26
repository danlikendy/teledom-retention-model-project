# teledom-retention-model-project

Прогноз оттока абонентов телеком-оператора «ТелеДом». Выпускной проект Data Scientist.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.912-success.svg)](#результаты)

## Описание

Модель бинарной классификации предсказывает, разорвёт ли абонент договор. Данные — SQLite-база с договорами, персональной информацией, интернет- и телефонными услугами (7043 клиента, срез на 01.02.2020).

**Задача заказчика:** заранее находить клиентов с риском оттока и предлагать промокоды / специальные условия.

## Результаты

| Метрика | Значение |
|---------|----------|
| ROC-AUC (test) | **0.976** |
| Accuracy (test) | **0.968** |
| Лучшая модель | Gradient Boosting + Optuna |

Ключевые факторы оттока: короткий срок договора, помесячная оплата, electronic check, высокий `MonthlyCharges`.

## Структура

```
├── data/                          # SQLite-база (скачивается отдельно)
├── notebooks/
│   └── telecom_churn_prediction.ipynb
├── scripts/
│   └── generate_notebook.py
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
git clone https://github.com/danlikendy/teledom-retention-model-project.git
cd teledom-retention-model-project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data
wget -O data/ds-plus-final.db https://code.s3.yandex.net/data-scientist/ds-plus-final.db

jupyter notebook notebooks/telecom_churn_prediction.ipynb
```

## Данные

| Таблица | Содержание |
|---------|------------|
| `contract` | договор, оплата, расходы |
| `personal` | пол, семья, возраст |
| `internet` | тип интернета и доп. услуги |
| `phone` | телефония |

Целевая переменная: `EndDate != 'No'` → отток (15.6%).

## Модели

- Decision Tree
- Random Forest + RandomizedSearchCV
- Gradient Boosting + Optuna (5 гиперпараметров)
- MLP (нейросеть)

`RANDOM_STATE = 250826`

## Автор

[danlikendy](https://github.com/danlikendy)
