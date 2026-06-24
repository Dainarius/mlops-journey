"""
Тесты для модуля инференса.

Проверяем:
1. Подготовку сырых данных
2. Предсказание модели
3. Формат ответа
"""

import pytest
import pandas as pd
from pathlib import Path

from src.predict import prepare_raw_data, predict_churn


class TestPrepareRawData:
    """Тесты для подготовки сырых данных."""

    def test_prepare_raw_data_from_dict(self):
        """Проверяем, что словарь преобразуется в DataFrame."""
        raw_data = {"customerID": "12345", "gender": "Male", "tenure": 12, "MonthlyCharges": 70.5}

        df = prepare_raw_data(raw_data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "gender" in df.columns

    def test_prepare_raw_data_removes_customerid(self):
        """Проверяем, что customerID удаляется."""
        raw_data = {"customerID": "12345", "gender": "Male", "tenure": 12}

        df = prepare_raw_data(raw_data)

        assert "customerID" not in df.columns

    def test_prepare_raw_data_removes_churn(self):
        """Проверяем, что Churn удаляется (если вдруг пришёл)."""
        raw_data = {"customerID": "12345", "gender": "Male", "tenure": 12, "Churn": "Yes"}

        df = prepare_raw_data(raw_data)

        assert "Churn" not in df.columns


class TestPredictChurn:
    """Тесты для функции предсказания."""

    @pytest.fixture
    def sample_customer(self):
        """Фикстура с тестовыми данными клиента."""
        return {
            "customerID": "TEST-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card",
            "MonthlyCharges": 65.5,
            "TotalCharges": "786.0",
        }

    def test_predict_churn_returns_dict(self, sample_customer):
        """Проверяем, что предсказание возвращает словарь."""
        pipeline_path = Path("artifacts/churn_pipeline.joblib")

        # Пропускаем тест, если модель не обучена
        if not pipeline_path.exists():
            pytest.skip("Pipeline не найден. Запустите обучение сначала.")

        result = predict_churn(sample_customer, pipeline_path)

        assert isinstance(result, dict)

    def test_predict_churn_contains_required_fields(self, sample_customer):
        """Проверяем, что результат содержит все необходимые поля."""
        pipeline_path = Path("artifacts/churn_pipeline.joblib")

        if not pipeline_path.exists():
            pytest.skip("Pipeline не найден.")

        result = predict_churn(sample_customer, pipeline_path)

        required_fields = ["customer_id", "prediction", "churn_probability", "risk_level"]
        for field in required_fields:
            assert field in result, f"Отсутствует поле: {field}"

    def test_predict_churn_probability_range(self, sample_customer):
        """Проверяем, что вероятность в диапазоне [0, 1]."""
        pipeline_path = Path("artifacts/churn_pipeline.joblib")

        if not pipeline_path.exists():
            pytest.skip("Pipeline не найден.")

        result = predict_churn(sample_customer, pipeline_path)

        assert 0 <= result["churn_probability"] <= 1

    def test_predict_churn_prediction_values(self, sample_customer):
        """Проверяем, что предсказание — это 'Yes' или 'No'."""
        pipeline_path = Path("artifacts/churn_pipeline.joblib")

        if not pipeline_path.exists():
            pytest.skip("Pipeline не найден.")

        result = predict_churn(sample_customer, pipeline_path)

        assert result["prediction"] in ["Yes", "No"]

    def test_predict_churn_risk_level_values(self, sample_customer):
        """Проверяем, что уровень риска — это 'HIGH', 'MEDIUM' или 'LOW'."""
        pipeline_path = Path("artifacts/churn_pipeline.joblib")

        if not pipeline_path.exists():
            pytest.skip("Pipeline не найден.")

        result = predict_churn(sample_customer, pipeline_path)

        assert result["risk_level"] in ["HIGH", "MEDIUM", "LOW"]
