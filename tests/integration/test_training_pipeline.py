from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.training.dataset import create_training_dataset

FIXTURE_PATH = Path("tests/fixtures/events.csv")


def test_training_pipeline() -> None:
    """Raw events can be transformed into model-ready examples."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)

    dataset = create_training_dataset(features)
    assert len(dataset) == 5

    targets = {example.target for example in dataset}
    assert targets == {0, 1}
