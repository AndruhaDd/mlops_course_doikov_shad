import pandas as pd
from sklearn.model_selection import train_test_split

# Те признаки, которые мы реально используем в модели.
# Они присутствуют в датасете city_day.csv и в processed версиях.
FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Читает датасет из CSV."""
    return pd.read_csv(path)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Препроцессинг, одинаковый для train и inference.

    Важно: мы НЕ делаем pd.get_dummies() на City/Date, потому что:
    - Date даст тысячи уникальных значений и «взорвёт» количество фичей,
    - City и AQI_Bucket здесь не нужны для базовой регрессии.

    Поэтому оставляем только 12 числовых признаков загрязнителей.
    """

    # Берём только нужные колонки (если лишнее — выкидываем)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    # Приводим к числам, мусор → NaN
    for c in FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Убираем строки с пропусками
    df = df.dropna().reset_index(drop=True)
    return df


def split_xy(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Разделяет на train/test, возвращает X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
