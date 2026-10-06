from ml_platform.model.baseline import ModelInput, train_baseline_model
from ml_platform.service.prediction import PredictionService
from ml_platform.training.dataset import TrainingExample


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for service tests."""
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


def test_prediction_service_returns_prediction() -> None:
    """The prediction service delegates to the inference layer."""
    model = train_baseline_model(make_examples())
    service = PredictionService(model)

    model_input = ModelInput(
        event_count=12,
        session_count=2,
        page_view_count=7,
        feature_usage_count=2,
    )

    result = service.predict(model_input)

    assert result.predicted_class in {0, 1}
    assert 0.0 <= result.probability <= 1.0
