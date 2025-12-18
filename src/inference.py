import json
from pathlib import Path

import pandas as pd

from src.preprocess import FEATURE_COLUMNS, preprocess_features
from src.utils import load_model


def load_features(features_path: str) -> list[str]:
    p = Path(features_path)
    return json.loads(p.read_text(encoding="utf-8"))


def predict_one(model, raw_data: dict) -> float:
    """Предсказывает AQI по одному объекту."""
    df = pd.DataFrame([raw_data])
    df = preprocess_features(df)
    return float(model.predict(df)[0])


if __name__ == "__main__":
    model = load_model("models/model.joblib")

    sample = {
        "PM2.5": 81.40,
        "PM10": 124.50,
        "NO": 1.44,
        "NO2": 20.50,
        "NOx": 12.08,
        "NH3": 10.72,
        "CO": 0.12,
        "SO2": 15.24,
        "O3": 127.09,
        "Benzene": 0.20,
        "Toluene": 6.50,
        "Xylene": 0.06,
    }

    print("FEATURES:", FEATURE_COLUMNS)
    print("PREDICTION:", predict_one(model, sample))
