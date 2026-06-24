"""
Препроцессинг данных для модели предсказания оттока клиентов.

Этот модуль содержит:
- Pipeline для трансформации сырых данных в признаки модели
- Функции для сохранения/загрузки pipeline как артефакта
- Все преобразования основаны на выводах из EDA (notebooks/01_eda.ipynb)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# КОНСТАНТЫ: определения колонок на основе EDA
# ═══════════════════════════════════════════════════════════════

# Колонки для удаления
DROP_COLUMNS = ["customerID"]

# Числовые признаки
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Бинарные категориальные признаки (2 значения)
BINARY_CATEGORICAL = [
    "gender",
    "SeniorCitizen",  # Уже 0/1, но оставим как категорию
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",  # 3 значения, но обработаем отдельно
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",  # 3 значения
    "PaperlessBilling",
    "PaymentMethod",  # 4 значения
]

# Мульти-категориальные признаки (>2 значений)
MULTI_CATEGORICAL = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]

# Целевая переменная
TARGET_COLUMN = "Churn"


# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ: загрузка и подготовка данных
# ═══════════════════════════════════════════════════════════════


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Загрузка данных из CSV.

    Args:
        filepath: путь к CSV файлу

    Returns:
        DataFrame с данными
    """
    logger.info(f"Загрузка данных из {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка данных (до pipeline).

    Включает:
    - Удаление идентификаторов
    - Конвертацию TotalCharges в число
    - Обработку пропусков
    """
    logger.info("Начало очистки данных")

    # Копируем, чтобы не менять оригинал
    df_clean = df.copy()

    # Удаляем customerID
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
        logger.info("Удалена колонка customerID")

    # Конвертируем TotalCharges в число (была строкой!)
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        logger.info("TotalCharges конвертирован в числовой тип")

    # Заполняем пропуски в TotalCharges медианой
    if df_clean["TotalCharges"].isnull().sum() > 0:
        median_value = df_clean["TotalCharges"].median()
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_value)
        logger.info(f"Заполнено {df_clean['TotalCharges'].isnull().sum()} пропусков в TotalCharges")

    logger.info(f"Очистка завершена. Размер: {df_clean.shape}")
    return df_clean


# ═══════════════════════════════════════════════════════════════
# PIPELINE: создание препроцессинг-пайплайна
# ═══════════════════════════════════════════════════════════════


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Создаёт ColumnTransformer для обработки разных типов признаков.

    Returns:
        ColumnTransformer с пайплайнами для числовых и категориальных признаков
    """
    logger.info("Создание препроцессинг-пайплайна")

    # Числовые признаки: импутер + масштабирование
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    # Бинарные категориальные признаки: one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer применяет разные трансформеры к разным колонкам
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, MULTI_CATEGORICAL),
        ],
        remainder="drop",  # Остальные колонки (бинарные) удаляем
    )

    logger.info(f"Пайплайн создан. Числовые: {NUMERIC_FEATURES}, Категориальные: {MULTI_CATEGORICAL}")

    return preprocessor


# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ: применение pipeline и сохранение артефактов
# ═══════════════════════════════════════════════════════════════


def fit_and_transform(df: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применяет pipeline к данным и возвращает признаки и целевую переменную.

    Args:
        df: очищенный DataFrame
        preprocessor: ColumnTransformer

    Returns:
        X: массив признаков (numpy array)
        y: массив целевой переменной (numpy array)
    """
    logger.info("Применение препроцессинг-пайплайна")

    # Разделяем признаки и целевую переменную
    X = df.drop(columns=[TARGET_COLUMN])
    y = (df[TARGET_COLUMN] == "Yes").astype(int)  # Yes=1, No=0

    # Применяем трансформации
    X_processed = preprocessor.fit_transform(X)

    logger.info(f"Обработано признаков: {X_processed.shape[1]}")
    logger.info(f"Размер X: {X_processed.shape}, размер y: {y.shape}")

    return X_processed, y


def save_artifact(artifact, filepath: Path, name: str = "артефакт") -> None:
    """
    Сохраняет артефакт (pipeline, модель, scaler) в файл.

    Args:
        artifact: объект для сохранения
        filepath: путь к файлу
        name: название для логирования
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, filepath)
    logger.info(f"💾 {name.capitalize()} сохранён: {filepath}")


def load_artifact(filepath: Path, name: str = "артефакт"):
    """
    Загружает артефакт из файла.

    Args:
        filepath: путь к файлу
        name: название для логирования

    Returns:
        Загруженный объект
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{name.capitalize()} не найден: {filepath}")

    artifact = joblib.load(filepath)
    logger.info(f"📂 {name.capitalize()} загружен: {filepath}")

    return artifact


# ═══════════════════════════════════════════════════════════════
# MAIN: тестирование препроцессинга
# ═══════════════════════════════════════════════════════════════


def main():
    """
    Тестовый запуск препроцессинга.
    """
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПРЕПРОЦЕССИНГА")
    logger.info("=" * 60)

    # Пути
    data_path = Path("data/raw/churn.csv")
    artifact_path = Path("artifacts/preprocessor.joblib")

    # 1. Загрузка данных
    df_raw = load_data(data_path)

    # 2. Очистка данных
    df_clean = clean_data(df_raw)

    # 3. Создание pipeline
    preprocessor = create_preprocessing_pipeline()

    # 4. Применение pipeline
    X, y = fit_and_transform(df_clean, preprocessor)

    # 5. Сохранение pipeline как артефакта
    save_artifact(preprocessor, artifact_path, "preprocessor")

    # 6. Проверка: загружаем pipeline и применяем к новым данным
    logger.info("\n🧪 Проверка: загрузка pipeline и применение к первым 5 строкам")
    preprocessor_loaded = load_artifact(artifact_path, "preprocessor")
    X_sample = preprocessor_loaded.transform(df_clean.head(5).drop(columns=[TARGET_COLUMN]))
    logger.info(f"Размер обработанной выборки: {X_sample.shape}")
    logger.info(f"Первая строка: {X_sample[0, :5]}...")  # Первые 5 признаков

    logger.info("\n✅ ПРЕПРОЦЕССИНГ ЗАВЕРШЁН УСПЕШНО")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
