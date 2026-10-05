.PHONY: format lint typecheck test check

format:
	ruff format .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

check:
	ruff format --check .
	ruff check .
	mypy src
	pytest
