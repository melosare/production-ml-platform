FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
COPY artifacts ./artifacts

RUN pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "ml_platform.api.main:create_application", "--factory", "--host", "0.0.0.0", "--port", "8000"]
