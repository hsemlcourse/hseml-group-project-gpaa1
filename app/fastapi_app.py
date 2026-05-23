import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь, чтобы импортировать src
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import clean_data, add_features

# Загрузка модели (путь относительно корня проекта)
model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"
model = joblib.load(model_path)

app = FastAPI(title="Customer Churn Prediction API")

class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    try:
        input_df = pd.DataFrame([customer.dict()])
        input_df = clean_data(input_df)
        input_df = add_features(input_df)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return {
            "churn_prediction": int(prediction),       # преобразуем в int
            "churn_probability": float(probability)    # преобразуем в float
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}
