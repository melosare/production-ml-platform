from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.training.dataset import create_training_dataset

FIXTURE_PATH = "tests/fixtures/synthetic_events.csv"


def test_synthetic_dataset_builds_training_dataset() -> None:
    """Synthetic events produce a deterministic training dataset."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    assert len(examples) == 100
    assert {example.target for example in examples} == {0, 1}
