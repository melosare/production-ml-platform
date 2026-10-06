from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.training.dataset import create_training_dataset
from ml_platform.training.pipeline import train_and_persist_model
from ml_platform.training.split import split_training_dataset

FIXTURE_PATH = Path("tests/fixtures/synthetic_events.csv")


def test_training_pipeline() -> None:
    """Raw events can be transformed into model-ready examples."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)

    dataset = create_training_dataset(features)

    assert len(dataset) == 100

    targets = {example.target for example in dataset}
    assert targets == {0, 1}


def test_training_pipeline_persists_model_and_metadata(
    tmp_path: Path,
) -> None:
    """The training lifecycle persists a model and its metadata."""
    model_path = tmp_path / "baseline_model.joblib"
    metadata_path = tmp_path / "baseline_model.metadata.json"

    metadata = train_and_persist_model(
        data_path=FIXTURE_PATH,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert model_path.exists()
    assert metadata_path.exists()

    assert metadata.model_version == "baseline-1.0.0"
    assert metadata.model_type == "LogisticRegression"

    assert metadata.feature_names == [
        "event_count",
        "session_count",
        "page_view_count",
        "feature_usage_count",
    ]

    assert metadata.random_state == 42

    assert metadata.training_examples == 70
    assert metadata.validation_examples == 10
    assert metadata.test_examples == 20

    assert set(metadata.validation_metrics) == {
        "accuracy",
        "precision",
        "recall",
    }
    assert set(metadata.test_metrics) == {
        "accuracy",
        "precision",
        "recall",
    }

    for metrics in (
        metadata.validation_metrics,
        metadata.test_metrics,
    ):
        assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_training_split_is_reproducible() -> None:
    """The training lifecycle uses a deterministic split."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    first = split_training_dataset(
        examples,
        random_state=42,
    )
    second = split_training_dataset(
        examples,
        random_state=42,
    )

    assert first == second
