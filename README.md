[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание оттока клиента

**Студент:** Петросян Гурген Аликович

**Группа:** БИВ 237


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
   - [Установка окружения](#установка-окружения)
   - [Воспроизведение экспериментов (Jupyter)](#воспроизведение-экспериментов-jupyter)
   - [Запуск деплоя](#запуск-деплоя)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Деплой](#деплой)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Бинарная классификация – предсказание оттока клиентов банка (churn prediction) на основе анкетных данных.

**Датасет:** [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) (Kaggle).  
Содержит 10 000 строк, 14 признаков (кредитный рейтинг, страна, пол, возраст, стаж, баланс, количество продуктов, наличие карты, активность, зарплата и др.). Целевая переменная `Exited` (1 – клиент ушёл, 0 – остался).

**Целевая метрика:** **F1-score** (из-за дисбаланса классов). Дополнительно оценивались Accuracy, Precision, Recall, ROC-AUC.


## Структура репозитория
```
.
├── app
│ ├── fastapi_app.py # API сервер (FastAPI)
│ └── streamlit_app.py # Веб‑интерфейс (Streamlit)
├── data
│ ├── processed # Очищенные и обработанные данные
│ └── raw # Исходные файлы
├── models # Сохранённая модель best_model.pkl
├── notebooks
│ ├── 01_eda.ipynb # EDA, очистка, feature engineering
│ ├── 02_baseline.ipynb # Baseline-модель
│ └── 03_experiments.ipynb # Эксперименты и выбор лучшей модели
├── presentation # Презентация для защиты
├── report
│ ├── images # Изображения для отчёта
│ └── report.md # Финальный отчёт 
├── src
│ ├── preprocessing.py # Предобработка данных
│ └── modeling.py # Обучение и оценка моделей
├── tests
│ └── test.py # Тесты пайплайна
├── requirements.txt
└── README.md
```


## Быстрый старт

### Установка окружения

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-<ваш_логин>.git
cd hseml-group-project-<ваш_логин>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

**Воспроизведение экспериментов (Jupyter)**

Все исследования находятся в папке notebooks/. Для повторения результатов выполните:

bash

_\# Запустить Jupyter (если не установлен, добавьте в requirements.txt или установите отдельно)_

jupyter notebook

Затем откройте и выполните ноутбуки в порядке нумерации:

1.  01_eda.ipynb – разведочный анализ, очистка, создание новых признаков, визуализации.  
    Сохраняет обработанный датасет в data/processed/churn_featured.csv.
2.  02_baseline.ipynb – обучение логистической регрессии (baseline), расчёт метрик на валидации.
3.  03_experiments.ipynb – обучение всех моделей (Random Forest, XGBoost и др.), подбор гиперпараметров, сравнение, сохранение лучшей модели в models/best_model.pkl.

**Запуск деплоя**

После того как модель сохранена, можно запустить API и веб‑интерфейс.

**FastAPI сервер (бэкенд):**

bash

uvicorn app.fastapi_app:app --reload --port 8000

**Streamlit интерфейс (фронтенд) – в другом терминале:**

bash

streamlit run app/streamlit_app.py

После этого откроется браузер с формой ввода данных клиента. Заполните поля и нажмите **Predict** – получите предсказание и вероятность оттока.

## Данные

- data/raw/Churn_Modelling.csv – исходный датасет (10 000 × 14).
- data/processed/churn_featured.csv – после очистки и feature engineering (создаётся в 01_eda.ipynb).

## Результаты

Лучшая модель – **XGBoost** (после подбора гиперпараметров).

| Модель | F1-score (val) | ROC‑AUC (val) | F1-score (test) |
| --- | --- | --- | --- |
| Logistic Regression | 0.499 | 0.771 | 0.49   |
| KNN | 0.503 | 0.802 | 0.50   |
| Decision Tree | 0.577 | 0.839 | 0.57   |
| Random Forest | 0.620 | 0.861 | 0.619   |
| Gradient Boosting | 0.598 | 0.866 | 0,598   |
| **XGBoost** | **0.630** | **0.864** | **0.63** |

## Деплой

- **FastAPI** – бэкенд: принимает POST‑запросы с признаками клиента, возвращает предсказание и вероятность оттока.
- **Streamlit** – фронтенд: веб‑форма для ввода данных и отображения результата.

Пример запроса к API:

bash

curl -X POST http://localhost:8000/predict \\

\-H "Content-Type: application/json" \\

\-d '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,"Tenure":5,"Balance":50000,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":100000}'

Пример ответа:

json

{"churn_prediction":0,"churn_probability":0.2345}

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
