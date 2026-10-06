from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_platform.api.app import (
    create_app,
    create_app_from_config,
    create_app_from_model_path,
)
from ml_platform.model.baseline import ModelInput, train_baseline_model
from ml_platform.model.persistence import save_model
from ml_platform.service.prediction import PredictionService
from ml_platform.training.dataset import TrainingExample


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for API tests."""
    return [
        TrainingExample(
            user_id=f"user_{index:03d}",
            event_count=10 + index,
            session_count=2,
            page_view_count=5 + index,
            feature_usage_count=2,
            target=index % 2,
        )
        for index in range(10)
    ]


def create_test_client() -> TestClient:
    """Create a test client with a trained model."""
    model = train_baseline_model(make_examples())
    service = PredictionService(model)
    app = create_app(service)

    return TestClient(app)


def test_health_endpoint() -> None:
    """The health endpoint reports a healthy API."""
    client = create_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint() -> None:
    """The readiness endpoint reports a ready API."""
    client = create_test_client()

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_prediction_endpoint() -> None:
    """The prediction endpoint returns a model prediction."""
    client = create_test_client()

    response = client.post(
        "/predict",
        json={
            "event_count": 12,
            "session_count": 2,
            "page_view_count": 7,
            "feature_usage_count": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


@pytest.mark.parametrize(
    "field",
    [
        "event_count",
        "session_count",
        "page_view_count",
        "feature_usage_count",
    ],
)
def test_prediction_endpoint_rejects_negative_values(
    field: str,
) -> None:
    """Negative feature values are rejected by request validation."""
    client = create_test_client()

    payload = {
        "event_count": 12,
        "session_count": 2,
        "page_view_count": 7,
        "feature_usage_count": 2,
    }
    payload[field] = -1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_endpoint_rejects_missing_fields() -> None:
    """Missing required prediction fields are rejected."""
    client = create_test_client()

    response = client.post(
        "/predict",
        json={
            "event_count": 12,
            "session_count": 2,
            "page_view_count": 7,
        },
    )

    assert response.status_code == 422


def test_prediction_endpoint_returns_500_on_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction failures are returned as controlled server errors."""
    client = create_test_client()

    def failing_predict(_: ModelInput) -> object:
        raise RuntimeError("model inference failed")

    monkeypatch.setattr(
        PredictionService,
        "predict",
        failing_predict,
    )

    response = client.post(
        "/predict",
        json={
            "event_count": 12,
            "session_count": 2,
            "page_view_count": 7,
            "feature_usage_count": 2,
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "prediction failed"}


def test_create_app_from_model_path(tmp_path: Path) -> None:
    """The API can be created from a persisted model."""
    model = train_baseline_model(make_examples())
    model_path = tmp_path / "baseline_model.joblib"

    save_model(model, model_path)

    app = create_app_from_model_path(model_path)
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "event_count": 12,
            "session_count": 2,
            "page_view_count": 7,
            "feature_usage_count": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


def test_create_app_from_model_path_fails_when_model_is_missing(
    tmp_path: Path,
) -> None:
    """Application creation fails fast when the model is missing."""
    model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="model file does not exist"):
        create_app_from_model_path(model_path)


def test_create_app_from_config(tmp_path: Path) -> None:
    """The API can be created from application configuration."""
    model = train_baseline_model(make_examples())

    model_path = tmp_path / "baseline_model.joblib"
    save_model(model, model_path)

    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        f"environment: production\nlog_level: INFO\nmodel_path: {model_path}\n",
        encoding="utf-8",
    )

    app = create_app_from_config(config_path)
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "event_count": 12,
            "session_count": 2,
            "page_view_count": 7,
            "feature_usage_count": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


def test_create_app_from_config_supports_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables override YAML application settings."""
    model = train_baseline_model(make_examples())

    model_path = tmp_path / "baseline_model.joblib"
    save_model(model, model_path)

    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        "environment: development\n"
        "log_level: DEBUG\n"
        f"model_path: {tmp_path / 'missing_model.joblib'}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ML_PLATFORM_ENVIRONMENT", "production")
    monkeypatch.setenv("ML_PLATFORM_LOG_LEVEL", "INFO")
    monkeypatch.setenv("ML_PLATFORM_MODEL_PATH", str(model_path))

    app = create_app_from_config(config_path)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
