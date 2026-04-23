import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import RANDOM_STATE, build_preprocessor


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(n_neighbors=11),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.9,
            colsample_bytree=1.0,
            scale_pos_weight=3,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model, X_train):
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(model, X, y) -> dict:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    return metrics


def run_experiments(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    results = []
    models = get_models()

    for model_name, model in models.items():
        pipeline = build_pipeline(model, X_train)
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate_model(pipeline, X_val, y_val)

        row = {"model": model_name}
        row.update(val_metrics)
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)
    return results_df


def save_results(results_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

