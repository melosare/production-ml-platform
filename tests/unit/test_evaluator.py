import pytest

from ml_platform.evaluation.evaluator import evaluate_model
from ml_platform.training.dataset import TrainingExample


class FakeModel:
    """Deterministic fake classifier for evaluator tests."""

    def __init__(self, predictions: list[int]) -> None:
        self._predictions = predictions

    def predict(self, inputs: list[list[int]]) -> list[int]:
        """Return predetermined predictions."""
        return self._predictions


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for evaluator tests."""
    return [
        TrainingExample(
            user_id="user_001",
            event_count=3,
            session_count=1,
            page_view_count=2,
            feature_usage_count=1,
            target=0,
        ),
        TrainingExample(
            user_id="user_002",
            event_count=6,
            session_count=2,
            page_view_count=4,
            feature_usage_count=2,
            target=1,
        ),
        TrainingExample(
            user_id="user_003",
            event_count=7,
            session_count=2,
            page_view_count=5,
            feature_usage_count=2,
            target=1,
        ),
    ]


def test_evaluate_model() -> None:
    """A model is evaluated through the classification protocol."""
    model = FakeModel([0, 1, 0])

    result = evaluate_model(model, make_examples())

    assert result.metrics.accuracy == pytest.approx(2 / 3)
    assert result.metrics.precision == pytest.approx(1.0)
    assert result.metrics.recall == pytest.approx(0.5)


def test_evaluate_model_rejects_empty_examples() -> None:
    """An empty evaluation dataset is rejected."""
    model = FakeModel([])

    with pytest.raises(ValueError, match="examples must not be empty"):
        evaluate_model(model, [])
