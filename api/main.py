import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path

app = FastAPI(title="MLOps Inference Service", version="1.0.0")

MODEL = None
FEATURES = None


def load_model_from_mlflow():
    run_id = os.getenv("MODEL_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("MODEL_RUN_ID is empty")

    mlruns_root = Path("/app/mlruns")

    # Находим папку run: /app/mlruns/<exp_id>/<run_id>/
    run_dir = None
    for p in mlruns_root.glob("*/*"):
        if p.is_dir() and p.name == run_id:
            run_dir = p
            break
    if run_dir is None:
        raise RuntimeError(f"Run directory for run_id={run_id} not found under {mlruns_root}")

    artifacts_dir = run_dir / "artifacts"

    model_path = artifacts_dir / "model" / "model.pkl"
    features_path = artifacts_dir / "model_meta" / "features.json"

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise RuntimeError(f"Features not found: {features_path}")

    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    return model, features


def transform_payload_to_features(payload: dict, features: list[str]) -> pd.DataFrame:
    """
    Приводим вход (сырой JSON с City/Date/загрязнителями) к тем же фичам,
    на которых обучалась модель (FEATURES).
    """
    df = pd.DataFrame([payload])

    # Уберём то, что не должно участвовать в регрессии, если пришло
    if "AQI" in df.columns:
        df = df.drop(columns=["AQI"])
    if "AQI_Bucket" in df.columns:
        df = df.drop(columns=["AQI_Bucket"])

    # Date -> признаки
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["dayofweek"] = dt.dt.dayofweek
        df = df.drop(columns=["Date"])

    # One-hot для City (и любых строковых полей, если появятся)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Добавляем отсутствующие колонки как 0
    for c in features:
        if c not in df.columns:
            df[c] = 0

    # Отбрасываем лишние колонки (если пришли неожиданные поля)
    df = df[features]

    # Если где-то NaN (например, Date не распарсился) — делаем 0
    df = df.fillna(0)

    return df


@app.on_event("startup")
def _startup():
    global MODEL, FEATURES
    MODEL, FEATURES = load_model_from_mlflow()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    if MODEL is None or FEATURES is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = transform_payload_to_features(payload, FEATURES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")

    pred = MODEL.predict(X)[0]
    return {"prediction": float(pred)}
