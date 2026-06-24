"""
Тесты для валидации данных.

Проверяем:
1. Схему данных (колонки, типы)
2. Диапазоны значений
3. Отсутствие критических пропусков
"""

import pytest
import pandas as pd
from pathlib import Path


class TestDataSchema:
    """Тесты для проверки схемы данных."""

    @pytest.fixture
    def churn_data(self):
        """Фикстура: загружаем данные."""
        data_path = Path("data/raw/churn.csv")
        if not data_path.exists():
            pytest.skip("Файл данных не найден.")
        return pd.read_csv(data_path)

    def test_data_has_expected_columns(self, churn_data):
        """Проверяем, что есть все ожидаемые колонки."""
        expected_columns = [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

        for col in expected_columns:
            assert col in churn_data.columns, f"Отсутствует колонка: {col}"

    def test_data_has_reasonable_size(self, churn_data):
        """Проверяем, что размер данных разумный."""
        assert len(churn_data) > 1000, "Слишком мало данных"
        assert len(churn_data) < 100000, "Слишком много данных"

    def test_churn_column_values(self, churn_data):
        """Проверяем, что Churn содержит только 'Yes' и 'No'."""
        unique_values = set(churn_data["Churn"].unique())
        assert unique_values == {"Yes", "No"}


class TestDataQuality:
    """Тесты для проверки качества данных."""

    @pytest.fixture
    def churn_data(self):
        """Фикстура: загружаем данные."""
        data_path = Path("data/raw/churn.csv")
        if not data_path.exists():
            pytest.skip("Файл данных не найден.")
        return pd.read_csv(data_path)

    def test_tenure_range(self, churn_data):
        """Проверяем, что tenure в разумном диапазоне."""
        assert churn_data["tenure"].min() >= 0
        assert churn_data["tenure"].max() <= 100

    def test_monthly_charges_range(self, churn_data):
        """Проверяем, что MonthlyCharges в разумном диапазоне."""
        assert churn_data["MonthlyCharges"].min() >= 0
        assert churn_data["MonthlyCharges"].max() <= 200

    def test_no_duplicate_customer_ids(self, churn_data):
        """Проверяем, что нет дубликатов customerID."""
        assert churn_data["customerID"].nunique() == len(churn_data)

    def test_critical_columns_not_empty(self, churn_data):
        """Проверяем, что критические колонки не пустые."""
        critical_columns = ["tenure", "MonthlyCharges", "Churn"]
        for col in critical_columns:
            assert churn_data[col].notna().sum() > 0.95 * len(churn_data)
