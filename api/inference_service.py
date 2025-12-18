import os
from pathlib import Path
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv

from src.preprocess import FEATURE_COLUMNS, preprocess_features
from src.utils import load_model, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Загружаем переменные окружения (docker-compose использует env_file configs/.env)
load_dotenv(str(PROJECT_ROOT / "configs" / ".env"))


def _map_request_to_dataset_columns(payload: dict) -> dict:
    """Маппит поля из API формата в названия колонок датасета."""
    mapping = {
        "PM2_5": "PM2.5",
    }

    mapped = {}
    for k, v in payload.items():
        mapped[mapping.get(k, k)] = v
    return mapped


@lru_cache(maxsize=1)
def get_settings() -> dict:
    """Берём настройки из YAML, а пути можно переопределить через .env."""
    cfg = load_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))
    inf = cfg.get("inference", {})

    # приоритет у переменных окружения
    model_path = os.getenv("MODEL_PATH", inf.get("model_path", "models/model.joblib"))
    features_path = os.getenv("FEATURES_PATH", inf.get("features_path", "models/features.json"))

    # Если пути относительные — считаем их относительно корня проекта
    inf["model_path"] = str((PROJECT_ROOT / model_path).resolve()) if not os.path.isabs(model_path) else model_path
    inf["features_path"] = str((PROJECT_ROOT / features_path).resolve()) if not os.path.isabs(features_path) else features_path
    return inf


@lru_cache(maxsize=1)
def get_model():
    settings = get_settings()
    model_path = settings["model_path"]
    return load_model(model_path)


def predict_from_json(payload: dict) -> float:
    model = get_model()
    mapped = _map_request_to_dataset_columns(payload)
    df = pd.DataFrame([mapped])
    df = preprocess_features(df)
    pred = model.predict(df)[0]
    return float(pred)
