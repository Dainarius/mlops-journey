.PHONY: setup train predict test clean

# Установка окружения
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Обучение модели
train:
	.venv/bin/python src/train.py
	
# Обучение с тюнингом гиперпараметров (пример)
train-tune:
	.venv/bin/python src/train.py --n_estimators 200 --max_depth 7	

# Предсказание
predict:
	.venv/bin/python src/predict.py

# Запуск тестов
test:
	.venv/bin/pytest tests/

# Очистка
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .venv mlruns/
