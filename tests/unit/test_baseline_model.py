import pytest

from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.model.baseline import (
    FEATURE_NAMES,
    to_model_input,
    train_baseline_model,
)
from ml_platform.training.dataset import create_training_dataset

FIXTURE_PATH = "tests/fixtures/events.csv"


def get_training_examples():
    """Build training examples from the event fixture."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)

    return create_training_dataset(features)


def test_model_input_contains_expected_features() -> None:
    """Model input contains only approved feature columns."""
    examples = get_training_examples()

    model_input = to_model_input(examples[0])

    assert len(FEATURE_NAMES) == 4
    assert model_input.event_count == examples[0].event_count
    assert model_input.session_count == examples[0].session_count
    assert model_input.page_view_count == examples[0].page_view_count
    assert model_input.feature_usage_count == examples[0].feature_usage_count


def test_train_baseline_model() -> None:
    """The baseline model can be trained."""
    examples = get_training_examples()

    model = train_baseline_model(examples)

    assert model is not None
    assert model.classes_.tolist() == [0, 1]


def test_train_baseline_model_rejects_empty_dataset() -> None:
    """An empty training dataset is rejected."""
    with pytest.raises(ValueError, match="examples must not be empty"):
        train_baseline_model([])
