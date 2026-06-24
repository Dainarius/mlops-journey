"""
Скрипт для инференса (предсказания) модели оттока клиентов.

MLOps-принципы:
1. Использование единого Pipeline (Preprocessor + Model) для инференса.
2. Приём "сырых" данных (имитация JSON-запроса от API).
3. Возврат как класса предсказания, так и вероятности для бизнес-логики.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

# Импортируем функцию загрузки артефакта из preprocess.py
from src.preprocess import load_artifact

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ: загрузка и подготовка данных
# ═══════════════════════════════════════════════════════════════


def load_pipeline(pipeline_path: Path = Path("artifacts/churn_pipeline.joblib")):
    """
    Загружает обученный pipeline из файла.

    Args:
        pipeline_path: путь к файлу pipeline

    Returns:
        Загруженный sklearn Pipeline
    """
    return load_artifact(pipeline_path, "pipeline")


def prepare_raw_data(raw_data: dict) -> pd.DataFrame:
    """
    Преобразует сырые данные (dict/JSON) в DataFrame для инференса.

    Args:
        raw_data: словарь с данными клиента

    Returns:
        DataFrame с одной строкой, готовый для pipeline
    """
    logger.info("Преобразование сырых данных в DataFrame...")

    # Преобразуем словарь в DataFrame (одна строка)
    df_raw = pd.DataFrame([raw_data])

    # Защита: если вдруг в JSON прилетели customerID или Churn, удаляем их
    cols_to_drop = [col for col in ["customerID", "Churn"] if col in df_raw.columns]
    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)
        logger.info(f"Удалены служебные колонки: {cols_to_drop}")

    return df_raw


def predict_churn(raw_data: dict, pipeline_path: Path = Path("artifacts/churn_pipeline.joblib")) -> dict:
    """
    Основная функция предсказания.

    Args:
        raw_data: Словарь с сырыми данными клиента.
        pipeline_path: Путь к сохранённому пайплайну.

    Returns:
        Словарь с результатом предсказания.
    """
    logger.info("=" * 60)
    logger.info("ЗАПУСК ИНФЕРЕНСА")
    logger.info("=" * 60)

    # 1. Загрузка пайплайна
    pipeline = load_pipeline(pipeline_path)

    # 2. Подготовка данных
    df_ready = prepare_raw_data(raw_data)

    # 3. Предсказание
    # pipeline.predict вернет массив, берем первый элемент [0]
    prediction_class = pipeline.predict(df_ready)[0]

    # pipeline.predict_proba вернет массив вероятностей [[prob_No, prob_Yes]]
    # Нам нужна вероятность класса "Yes" (индекс 1)
    probabilities = pipeline.predict_proba(df_ready)[0]
    prob_churn = probabilities[1]

    # 4. Формирование ответа
    result = {
        "customer_id": raw_data.get("customerID", "unknown"),
        "prediction": "Yes" if prediction_class == 1 else "No",
        "churn_probability": round(float(prob_churn), 4),
        "risk_level": "HIGH" if prob_churn > 0.7 else ("MEDIUM" if prob_churn > 0.4 else "LOW"),
    }

    logger.info(f"✅ Предсказание готово: {result}")
    logger.info("=" * 60)

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN: Тестирование инференса
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Имитация "сырого" JSON-запроса от фронтенда или CRM-системы
    # Обратите внимание: TotalCharges передан как строка (как в исходном CSV!),
    # а MonthlyCharges как float. Пайплайн должен справиться с этим.
    sample_customer = {
        "customerID": "7590-VHVEG",  # Будет проигнорирован пайплайном
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": "70.70",  # Специально строка, чтобы проверить robustness
    }

    # Запуск предсказания
    result = predict_churn(sample_customer)

    # Красивый вывод в консоль (имитация ответа API)
    print("\n📡 Ответ API (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
