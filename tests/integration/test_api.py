from pathlib import Path

from fastapi.testclient import TestClient

from ml_platform.api.app import create_app, create_app_from_model_path
from ml_platform.model.baseline import train_baseline_model
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
