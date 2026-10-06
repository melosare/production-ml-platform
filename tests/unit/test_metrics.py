import pytest

from ml_platform.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)


def test_calculate_classification_metrics() -> None:
    """Classification metrics are calculated correctly."""
    targets = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]

    metrics = calculate_classification_metrics(targets, predictions)

    assert metrics == ClassificationMetrics(
        accuracy=0.75,
        precision=pytest.approx(2 / 3),
        recall=1.0,
    )


def test_calculate_classification_metrics_rejects_empty_targets() -> None:
    """Empty targets are rejected."""
    with pytest.raises(ValueError, match="targets must not be empty"):
        calculate_classification_metrics([], [])


def test_calculate_classification_metrics_rejects_mismatched_lengths() -> None:
    """Targets and predictions must have matching lengths."""
    with pytest.raises(
        ValueError,
        match="targets and predictions must have the same length",
    ):
        calculate_classification_metrics([0, 1], [0])
