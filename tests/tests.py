from pathlib import Path

from src.preprocessing import add_features, clean_data, load_data, split_data

DATA_PATH = Path("data/raw/Churn_Modelling.csv")


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_load_data():
    df = load_data(DATA_PATH)
    assert df.shape[0] == 10000
    assert "Exited" in df.columns


def test_clean_data_removes_unused_columns():
    df = load_data(DATA_PATH)
    cleaned = clean_data(df)
    assert "RowNumber" not in cleaned.columns
    assert "CustomerId" not in cleaned.columns
    assert "Surname" not in cleaned.columns


def test_feature_engineering_adds_columns():
    df = load_data(DATA_PATH)
    cleaned = clean_data(df)
    featured = add_features(cleaned)
    assert "BalanceSalaryRatio" in featured.columns
    assert "TenureAgeRatio" in featured.columns
    assert "IsSenior" in featured.columns


def test_split_data():
    df = load_data(DATA_PATH)
    cleaned = clean_data(df)
    featured = add_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(featured)
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
