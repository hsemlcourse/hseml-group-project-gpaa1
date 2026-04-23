from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)
    df["TenureAgeRatio"] = df["Tenure"] / (df["Age"] + 1e-6)
    df["IsSenior"] = (df["Age"] >= 45).astype(int)
    df["CreditScoreBucket"] = pd.cut(
        df["CreditScore"],
        bins=[0, 500, 650, 750, 1000],
        labels=["low", "medium", "good", "excellent"],
        include_lowest=True,
    )
    df["ProductsPerBalance"] = df["NumOfProducts"] / (df["Balance"] + 1)
    df["IsLowCredit"] = (df["CreditScore"] < 600).astype(int)
    df["HighBalance"] = (df["Balance"] > 100000).astype(int)
    df["AgeBucket"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100])
    return df


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numerical_features:
        numerical_features.remove(TARGET_COLUMN)
    return categorical_features, numerical_features


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features, numerical_features = get_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

