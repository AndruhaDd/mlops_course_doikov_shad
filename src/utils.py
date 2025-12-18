import pandas as pd
import numpy as np


def load_yaml(path: str):
    """Безопасно загружает YAML-конфиг."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_model(model, path: str):
    """Сохраняет модель через joblib."""
    import joblib
    joblib.dump(model, path)


def load_model(path: str):
    """Загружает модель."""
    import joblib
    return joblib.load(path)


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает строки с пропусками (на всякий случай)."""
    return df.dropna()


def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Превращает нужные колонки в числовые типы.
    Если внутри мусор — ставит NaN.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def enforce_column_order(df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """
    Переставляет фичи так же, как было при обучении.
    Это критично для корректного inference.
    """
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0  # добавляем отсутствующие колонки

    return df[reference_columns]
