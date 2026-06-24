"""
Обучение модели предсказания оттока клиентов с трекингом экспериментов в MLflow.

MLOps-принципы:
1. Автоматическое логирование гиперпараметров, метрик и артефактов.
2. Воспроизводимость: каждый запуск фиксируется с точностью до random_state.
3. Сравнение экспериментов через MLflow UI.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Импортируем наши функции из preprocess.py
from src.preprocess import load_data, clean_data, create_preprocessing_pipeline, save_artifact

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки для гибкости обучения."""
    parser = argparse.ArgumentParser(description="Обучение модели Churn Prediction")
    parser.add_argument("--n_estimators", type=int, default=100, help="Количество деревьев в лесу")
    parser.add_argument("--max_depth", type=int, default=5, help="Максимальная глубина дерева")
    parser.add_argument("--test_size", type=float, default=0.2, help="Доля тестовой выборки")
    parser.add_argument("--random_state", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument(
        "--experiment_name", type=str, default="churn_prediction", help="Название эксперимента в MLflow"
    )
    return parser.parse_args()


def train_and_evaluate(args):
    """Основная функция обучения и оценки с логированием в MLflow."""
    logger.info("=" * 60)
    logger.info("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
    logger.info(f"Гиперпараметры: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # ИНИЦИАЛИЗАЦИЯ MLFLOW
    # ═══════════════════════════════════════════════════════════════
    # Устанавливаем tracking URI (локальное хранилище)
    # Используем абсолютный путь, чтобы избежать проблем с путями
    tracking_uri = Path("./mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"📂 MLflow tracking URI: {tracking_uri}")

    # Создаём или получаем эксперимент
    mlflow.set_experiment(args.experiment_name)

    # Начинаем новый run (запуск)
    with mlflow.start_run(run_name=f"rf_{args.n_estimators}est_{args.max_depth}depth"):
        logger.info(f"📊 MLflow Run ID: {mlflow.active_run().info.run_id}")

        # 1. Загрузка и очистка данных
        data_path = Path("data/raw/churn.csv")
        logger.info("Загрузка и очистка данных...")
        df_raw = load_data(data_path)
        df_clean = clean_data(df_raw)

        # 2. Разделение на признаки и целевую переменную
        X = df_clean.drop(columns=["Churn"])
        y = (df_clean["Churn"] == "Yes").astype(int)

        # 3. Разделение на train/test ДО фитинга препроцессора
        logger.info(f"Разделение данных (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        # ═══════════════════════════════════════════════════════════════
        # ЛОГИРОВАНИЕ ПАРАМЕТРОВ В MLFLOW
        # ═══════════════════════════════════════════════════════════════
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_samples", len(X_test))
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")

        # 4. Создание и фитинг препроцессора
        logger.info("Создание и фитинг препроцессинг-пайплайна...")
        preprocessor = create_preprocessing_pipeline()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # 5. Инициализация и обучение модели
        logger.info("Обучение модели RandomForest...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train_processed, y_train)

        # 6. Оценка качества на тестовой выборке
        logger.info("Оценка модели на тестовой выборке...")
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
        }

        logger.info("\n📊 Метрики модели:")
        for metric, value in metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")

        logger.info("\n📋 Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["No", "Yes"]))

        # ═══════════════════════════════════════════════════════════════
        # ЛОГИРОВАНИЕ МЕТИК В MLFLOW
        # ═══════════════════════════════════════════════════════════════
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        # 7. Создание финального пайплайна для инференса
        logger.info("Создание финального inference-пайплайна...")
        inference_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # 8. Сохранение артефактов
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Сохраняем пайплайн локально
        pipeline_path = artifacts_dir / "churn_pipeline.joblib"
        save_artifact(inference_pipeline, pipeline_path, "Inference Pipeline")

        # ═══════════════════════════════════════════════════════════════
        # ЛОГИРОВАНИЕ АРТЕФАКТОВ В MLFLOW
        # ═══════════════════════════════════════════════════════════════
        # Логируем сам pipeline как MLflow-артефакт
        mlflow.sklearn.log_model(
            sk_model=inference_pipeline, artifact_path="model", registered_model_name="churn_prediction_model"
        )

        # Логируем метрики в JSON
        metrics_path = artifacts_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=4)
        mlflow.log_artifact(str(metrics_path))

        # Логируем classification report
        report_path = artifacts_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
        mlflow.log_artifact(str(report_path))

        logger.info(f"💾 Метрики сохранены: {metrics_path}")
        logger.info(f"📦 Модель залогирована в MLflow")

        logger.info("=" * 60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        logger.info(f"🔗 MLflow Run ID: {mlflow.active_run().info.run_id}")
        logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
