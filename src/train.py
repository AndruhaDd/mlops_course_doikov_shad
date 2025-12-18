# src/train.py
import os
import json
from typing import Optional, Dict, Any, Tuple

import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def _ensure_dirs() -> None:
    """Создаём локальную папку для временных артефактов (не коммитится в git)."""
    os.makedirs("artifacts", exist_ok=True)


def _read_raw(data_path: str) -> pd.DataFrame:
    """Читаем исходный CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path}. "
            f"Укажи корректный путь или выставь DATA_PATH."
        )
    return pd.read_csv(data_path)


def _make_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Единая логика фичей (и для train, и для inference):
    - удаляем мусор
    - Date -> year/month/day/dayofweek
    - one-hot для City (и других object)
    - приводим к числам, NaN -> 0
    """
    df = df.copy()

    # Удаляем то, что точно не должно участвовать как фича
    if "AQI_Bucket" in df.columns:
        df = df.drop(columns=["AQI_Bucket"])

    # Проверяем наличие таргета
    if target_col not in df.columns:
        raise ValueError(
            f"Таргет-колонка '{target_col}' не найдена. "
            f"Колонки в датасете: {list(df.columns)}"
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Date -> признаки
    if "Date" in X.columns:
        dt = pd.to_datetime(X["Date"], errors="coerce")
        X["year"] = dt.dt.year
        X["month"] = dt.dt.month
        X["day"] = dt.dt.day
        X["dayofweek"] = dt.dt.dayofweek
        X = X.drop(columns=["Date"])

    # One-hot для строковых полей (City и т.п.)
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # Всё к числам
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(0)

    return X, y


def _apply_dataset_version(df: pd.DataFrame, dataset_version: str, target_col: str) -> pd.DataFrame:
    """
    Две версии датасета = два разных препроцессинга.
    Важно: в git мы это не доказываем файлами, а логируем результат в MLflow артефакты.
    """
    df = df.copy()

    # Базовая очистка: убираем пустые строки по ключевым полям
    # (оставляем только то, где есть таргет)
    df = df.dropna(subset=[target_col])

    if dataset_version == "v1":
        # v1: минимальная очистка
        # - dropna только по таргету (уже сделали)
        return df

    if dataset_version == "v2":
        # v2: чуть “жестче”:
        # - убираем выбросы по квантилям по основным загрязнителям (если есть)
        cols_for_clip = [
            "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
            "Benzene", "Toluene", "Xylene",
        ]
        present = [c for c in cols_for_clip if c in df.columns]
        for c in present:
            low, high = df[c].quantile([0.01, 0.99])
            df = df[(df[c] >= low) & (df[c] <= high)]
        return df

    raise ValueError("dataset_version должен быть 'v1' или 'v2'")


def _build_model(model_type: str, params: Dict[str, Any]):
    """Собираем модель по типу."""
    if model_type == "ridge":
        alpha = float(params.get("alpha", 1.0))
        return Ridge(alpha=alpha, random_state=int(params.get("random_state", 42)))

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    raise ValueError("model_type должен быть 'ridge' или 'rf'")


def train_one_run(
    data_path: str,
    dataset_version: str,
    model_type: str,
    random_state: int = 42,
    test_size: float = 0.2,
    alpha: float = 1.0,                 # для Ridge
    n_estimators: int = 300,            # для RF
    max_depth: Optional[int] = None,    # для RF
) -> str:
    """
    Один запуск обучения:
    - читает raw
    - делает версионный препроцессинг (v1/v2)
    - строит фичи
    - учит модель
    - логирует в MLflow: params/metrics/artifacts (dataset/model/features)
    - возвращает run_id
    """
    _ensure_dirs()

    target_col = os.getenv("TARGET_COL", "AQI")

    # MLflow: лучше всегда явно выставлять tracking uri
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("mlops-course-project")

    raw_df = _read_raw(data_path)

    # Применяем "версию датасета" (по факту — версию препроцессинга)
    df = _apply_dataset_version(raw_df, dataset_version=dataset_version, target_col=target_col)

    # Делаем X/y так же, как будет в инференсе
    X, y = _make_features(df, target_col=target_col)

    # Сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    # Параметры модели
    model_params: Dict[str, Any] = {
        "random_state": int(random_state),
        "test_size": float(test_size),
    }
    if model_type == "ridge":
        model_params["alpha"] = float(alpha)
    elif model_type == "rf":
        model_params["n_estimators"] = int(n_estimators)
        model_params["max_depth"] = None if max_depth is None else int(max_depth)

    model = _build_model(model_type=model_type, params=model_params)

    with mlflow.start_run() as run:
        # ===== params =====
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_col", target_col)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # ===== train =====
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # ===== metrics =====
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        # ===== artifacts: dataset (доказательство v1/v2) =====
        # Логируем небольшой, но воспроизводимый артефакт обработанного датасета
        # (можно и полный, но это тяжелее)
        processed_path = os.path.join("artifacts", f"processed_{dataset_version}.csv")
        df_to_log = df.copy()

        # Чтобы артефакт был компактнее — можно залогировать сэмпл
        # (и при этом это всё равно доказывает разные версии препроцессинга)
        if len(df_to_log) > 50000:
            df_to_log = df_to_log.sample(n=50000, random_state=int(random_state))

        df_to_log.to_csv(processed_path, index=False)
        mlflow.log_artifact(processed_path, artifact_path="dataset")

        # ===== artifacts: model =====
        model_pkl_path = os.path.join("artifacts", "model.pkl")
        joblib.dump(model, model_pkl_path)
        mlflow.log_artifact(model_pkl_path, artifact_path="model")

        # ===== artifacts: features for inference =====
        features = list(X.columns)
        features_path = os.path.join("artifacts", "features.json")
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(features_path, artifact_path="model_meta")

        return run.info.run_id
