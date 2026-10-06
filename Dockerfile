FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY docs ./docs

RUN pip install --no-cache-dir ".[dev]"

CMD ["pytest"]

