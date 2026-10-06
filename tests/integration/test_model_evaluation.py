from ml_platform.data.loader import load_events
from ml_platform.evaluation.evaluator import evaluate_model
from ml_platform.features.pipeline import build_features
from ml_platform.model.baseline import train_baseline_model
from ml_platform.training.dataset import create_training_dataset

FIXTURE_PATH = "tests/fixtures/synthetic_events.csv"


def test_baseline_model_evaluation() -> None:
    """The baseline model can be trained and evaluated."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    model = train_baseline_model(examples)
    result = evaluate_model(model, examples)

    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
