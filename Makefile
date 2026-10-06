.PHONY: test lint typecheck train run quality

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

train:
	python scripts/train_baseline_model.py

run:
	uvicorn ml_platform.api.main:create_application --factory

quality: lint typecheck test

