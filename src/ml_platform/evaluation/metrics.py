from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics for a binary model."""

    accuracy: float
    precision: float
    recall: float


def calculate_classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> ClassificationMetrics:
    """Calculate binary classification metrics."""
    if not targets:
        raise ValueError("targets must not be empty")

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")

    return ClassificationMetrics(
        accuracy=float(accuracy_score(targets, predictions)),
        precision=float(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
    )
