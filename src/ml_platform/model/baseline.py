from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from ml_platform.training.dataset import TrainingExample

FEATURE_NAMES = (
    "event_count",
    "session_count",
    "page_view_count",
    "feature_usage_count",
)


@dataclass(frozen=True)
class ModelInput:
    """Numeric model input for a single training example."""

    event_count: int
    session_count: int
    page_view_count: int
    feature_usage_count: int


def to_model_input(example: TrainingExample) -> ModelInput:
    """Convert a training example into model features."""
    return ModelInput(
        event_count=example.event_count,
        session_count=example.session_count,
        page_view_count=example.page_view_count,
        feature_usage_count=example.feature_usage_count,
    )


def train_baseline_model(
    examples: list[TrainingExample],
) -> LogisticRegression:
    """Train the baseline logistic regression model."""
    if not examples:
        raise ValueError("examples must not be empty")

    inputs = [to_model_input(example) for example in examples]

    x = [
        [
            item.event_count,
            item.session_count,
            item.page_view_count,
            item.feature_usage_count,
        ]
        for item in inputs
    ]

    y = [example.target for example in examples]
    model = LogisticRegression(random_state=42)
    model.fit(x, y)

    return model
