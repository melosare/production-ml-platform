from dataclasses import dataclass

from ml_platform.features.aggregations import UserFeatures


@dataclass(frozen=True)
class TrainingExample:
    """Model-ready training example."""

    user_id: str
    event_count: int
    session_count: int
    page_view_count: int
    feature_usage_count: int
    target: int


def create_training_example(features: UserFeatures) -> TrainingExample:
    """Convert aggregated user features into a training example."""
    target = int(features.purchase_count > 0)

    return TrainingExample(
        user_id=features.user_id,
        event_count=features.event_count,
        session_count=features.session_count,
        page_view_count=features.page_view_count,
        feature_usage_count=features.feature_usage_count,
        target=target,
    )


def create_training_dataset(
    features: list[UserFeatures],
) -> list[TrainingExample]:
    """Create training examples from user features."""
    return [create_training_example(feature) for feature in features]
