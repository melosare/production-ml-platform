from pathlib import Path

from fastapi.testclient import TestClient

from ml_platform.api.main import create_application
from ml_platform.model.baseline import train_baseline_model
from ml_platform.model.persistence import save_model
from ml_platform.training.dataset import TrainingExample


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for application tests."""
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


def test_main_app_health(tmp_path: Path) -> None:
    """The application entrypoint exposes a healthy API."""
    model = train_baseline_model(make_examples())
    model_path = tmp_path / "baseline_model.joblib"
    save_model(model, model_path)

    config_path = tmp_path / "development.yaml"
    config_path.write_text(
        f"environment: development\nlog_level: DEBUG\nmodel_path: {model_path}\n",
        encoding="utf-8",
    )

    app = create_application(config_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
