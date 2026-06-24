"""
Тесты для модуля препроцессинга.

Проверяем:
1. Загрузку и очистку данных
2. Работу препроцессинг-пайплайна
3. Сохранение и загрузку артефактов
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

from src.preprocess import (
    load_data,
    clean_data,
    create_preprocessing_pipeline,
    fit_and_transform,
    save_artifact,
    load_artifact,
)


class TestLoadData:
    """Тесты для функции загрузки данных."""

    def test_load_data_success(self):
        """Проверяем, что данные загружаются успешно."""
        data_path = Path("data/raw/churn.csv")
        df = load_data(data_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Churn" in df.columns

    def test_load_data_file_not_found(self):
        """Проверяем, что возникает ошибка при отсутствии файла."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent_file.csv"))


class TestLoadData:
    """Тесты для функции загрузки данных."""

    @pytest.fixture
    def data_path(self):
        """Путь к данным."""
        return Path("data/raw/churn.csv")

    def test_load_data_success(self, data_path):
        """Проверяем, что данные загружаются успешно."""
        # Пропускаем тест, если файла нет (например, в CI без данных)
        if not data_path.exists():
            pytest.skip(f"Файл данных не найден: {data_path}")

        df = load_data(data_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Churn" in df.columns

    def test_load_data_file_not_found(self):
        """Проверяем, что возникает ошибка при отсутствии файла."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent_file.csv"))


class TestPreprocessingPipeline:
    """Тесты для препроцессинг-пайплайна."""

    def test_pipeline_creates_correct_features(self):
        """Проверяем, что пайплайн создаёт ожидаемое количество признаков."""
        # Создаём тестовые данные
        df = pd.DataFrame(
            {
                "tenure": [1, 2, 3],
                "MonthlyCharges": [50.0, 60.0, 70.0],
                "TotalCharges": [50.0, 120.0, 210.0],
                "InternetService": ["DSL", "Fiber optic", "No"],
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaymentMethod": ["Electronic check", "Mailed check", "Credit card"],
                "Churn": ["Yes", "No", "No"],
            }
        )

        preprocessor = create_preprocessing_pipeline()
        X, y = fit_and_transform(df, preprocessor)

        # Проверяем размерность
        assert X.shape[0] == 3  # 3 строки
        assert X.shape[1] > 0  # Есть признаки
        assert len(y) == 3  # 3 целевые переменные

    def test_pipeline_handles_missing_values(self):
        """Проверяем, что пайплайн обрабатывает пропуски."""
        df = pd.DataFrame(
            {
                "tenure": [1, np.nan, 3],
                "MonthlyCharges": [50.0, 60.0, np.nan],
                "TotalCharges": [50.0, 120.0, 210.0],
                "InternetService": ["DSL", "Fiber optic", "No"],
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaymentMethod": ["Electronic check", "Mailed check", "Credit card"],
                "Churn": ["Yes", "No", "No"],
            }
        )

        preprocessor = create_preprocessing_pipeline()
        X, y = fit_and_transform(df, preprocessor)

        # Проверяем, что нет NaN в результате
        assert not np.isnan(X).any()

    def test_pipeline_consistency(self):
        """Проверяем, что пайплайн даёт одинаковый результат при повторном применении."""
        df = pd.DataFrame(
            {
                "tenure": [1, 2],
                "MonthlyCharges": [50.0, 60.0],
                "TotalCharges": [50.0, 120.0],
                "InternetService": ["DSL", "Fiber optic"],
                "Contract": ["Month-to-month", "One year"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "Churn": ["Yes", "No"],
            }
        )

        preprocessor = create_preprocessing_pipeline()
        X1, y1 = fit_and_transform(df, preprocessor)

        # Применяем тот же пайплайн к тем же данным
        X2 = preprocessor.transform(df.drop(columns=["Churn"]))

        # Результаты должны быть идентичны
        np.testing.assert_array_almost_equal(X1, X2)


class TestArtifactSaveLoad:
    """Тесты для сохранения и загрузки артефактов."""

    def test_save_and_load_artifact(self):
        """Проверяем, что артефакт сохраняется и загружается корректно."""
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Создаём тестовый объект
            test_obj = {"key": "value", "number": 42}

            # Сохраняем
            save_artifact(test_obj, tmp_path, "test artifact")

            # Загружаем
            loaded_obj = load_artifact(tmp_path, "test artifact")

            # Проверяем
            assert loaded_obj == test_obj
        finally:
            # Удаляем временный файл
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_artifact_not_found(self):
        """Проверяем, что возникает ошибка при отсутствии артефакта."""
        with pytest.raises(FileNotFoundError):
            load_artifact(Path("nonexistent.joblib"))
