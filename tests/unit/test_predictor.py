from ml_platform.inference.predictor import PredictionResult, predict
from ml_platform.model.baseline import ModelInput, train_baseline_model
from ml_platform.training.dataset import TrainingExample


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for inference tests."""
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


def test_predict_returns_prediction_result() -> None:
    """Inference returns a typed prediction result."""
    model = train_baseline_model(make_examples())

    model_input = ModelInput(
        event_count=12,
        session_count=2,
        page_view_count=7,
        feature_usage_count=2,
    )

    result = predict(model, model_input)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in {0, 1}
    assert 0.0 <= result.probability <= 1.0


def test_predict_probability_matches_predicted_class() -> None:
    """Prediction probability corresponds to the predicted class."""
    model = train_baseline_model(make_examples())

    model_input = ModelInput(
        event_count=12,
        session_count=2,
        page_view_count=7,
        feature_usage_count=2,
    )

    result = predict(model, model_input)

    if result.predicted_class == 1:
        assert result.probability >= 0.5
    else:
        assert result.probability < 0.5
