from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.evaluation.evaluator import evaluate_model
from ml_platform.features.pipeline import build_features
from ml_platform.model.baseline import train_baseline_model
from ml_platform.training.dataset import create_training_dataset
from ml_platform.training.split import split_training_dataset

FIXTURE_PATH = Path("tests/fixtures/synthetic_events.csv")


def test_baseline_model_evaluation_on_held_out_data() -> None:
    """The baseline model is evaluated on held-out users."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    split = split_training_dataset(examples)

    model = train_baseline_model(split.training)
    result = evaluate_model(model, split.test)

    assert len(split.training) == 70
    assert len(split.validation) == 10
    assert len(split.test) == 20

    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
