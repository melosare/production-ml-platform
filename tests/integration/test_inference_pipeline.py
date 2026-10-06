from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.inference.predictor import predict
from ml_platform.model.baseline import ModelInput, train_baseline_model
from ml_platform.model.persistence import load_model, save_model
from ml_platform.training.dataset import create_training_dataset
from ml_platform.training.split import split_training_dataset

FIXTURE_PATH = "tests/fixtures/synthetic_events.csv"


def test_model_lifecycle_from_training_to_inference(tmp_path: Path) -> None:
    """The trained model can be persisted and used for inference."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    split = split_training_dataset(examples)

    model = train_baseline_model(split.training)

    model_path = tmp_path / "models" / "baseline_model.joblib"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    example = split.test[0]

    model_input = ModelInput(
        event_count=example.event_count,
        session_count=example.session_count,
        page_view_count=example.page_view_count,
        feature_usage_count=example.feature_usage_count,
    )

    result = predict(loaded_model, model_input)

    assert result.predicted_class in {0, 1}
    assert 0.0 <= result.probability <= 1.0
