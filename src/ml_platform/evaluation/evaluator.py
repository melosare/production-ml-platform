from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ml_platform.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from ml_platform.training.dataset import TrainingExample


class ClassificationModel(Protocol):
    """Protocol for models that produce binary predictions."""

    def predict(self, inputs: Sequence[Sequence[int]]) -> Sequence[int]:
        """Predict binary targets for model inputs."""


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a classification model."""

    metrics: ClassificationMetrics


def evaluate_model(
    model: ClassificationModel,
    examples: list[TrainingExample],
) -> EvaluationResult:
    """Evaluate a classification model on examples."""
    if not examples:
        raise ValueError("examples must not be empty")

    inputs = [
        [
            example.event_count,
            example.session_count,
            example.page_view_count,
            example.feature_usage_count,
        ]
        for example in examples
    ]

    targets = [example.target for example in examples]
    predictions = model.predict(inputs)

    return EvaluationResult(
        metrics=calculate_classification_metrics(
            targets,
            list(predictions),
        )
    )
