"""Обучение модели предсказания оттока клиентов."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Для первого шага используем встроенный датасет
# Позже заменим на реальные данные о клиентах

def load_data():
    """Загрузка данных."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Обучение модели."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Оценка качества."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Accuracy: {acc:.4f}")
    print("\n📋 Classification report:")
    print(classification_report(y_test, y_pred))
    return acc

def save_model(model, accuracy):
    """Сохранение модели."""
    Path("models").mkdir(exist_ok=True)
    path = f"models/model_v1_acc{accuracy:.3f}.pkl"
    joblib.dump(model, path)
    print(f"💾 Модель сохранена: {path}")

def main():
    print("🚀 Запуск обучения...\n")
    X_train, X_test, y_train, y_test = load_data()
    print(f"📦 Обучающая выборка: {X_train.shape}")
    print(f"📦 Тестовая выборка: {X_test.shape}\n")

    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    save_model(model, accuracy)

    print("\n✅ Обучение завершено!")

if __name__ == "__main__":
    main()
