from ml_platform.features.aggregations import UserFeatures
from ml_platform.training.dataset import (
    create_training_dataset,
    create_training_example,
)


def make_features(
    *,
    user_id: str = "user_001",
    purchase_count: int = 0,
) -> UserFeatures:
    """Create a feature record for testing."""
    return UserFeatures(
        user_id=user_id,
        event_count=5,
        session_count=1,
        page_view_count=1,
        feature_usage_count=1,
        purchase_count=purchase_count,
        total_purchase_value=0.0,
    )


def test_create_training_example_without_purchase() -> None:
    """Users without purchases receive target 0."""
    features = make_features(purchase_count=0)

    example = create_training_example(features)

    assert example.user_id == "user_001"
    assert example.event_count == 5
    assert example.session_count == 1
    assert example.page_view_count == 1
    assert example.feature_usage_count == 1
    assert example.target == 0


def test_create_training_example_with_purchase() -> None:
    """Users with purchases receive target 1."""
    features = make_features(purchase_count=1)

    example = create_training_example(features)

    assert example.target == 1


def test_purchase_information_is_not_a_training_feature() -> None:
    """Purchase-derived values are excluded from the model inputs."""
    features = make_features(purchase_count=1)

    example = create_training_example(features)

    assert not hasattr(example, "purchase_count")
    assert not hasattr(example, "total_purchase_value")


def test_create_training_dataset() -> None:
    """A list of feature records becomes a list of training examples."""
    features = [
        make_features(user_id="user_001", purchase_count=0),
        make_features(user_id="user_002", purchase_count=1),
        make_features(user_id="user_003", purchase_count=0),
    ]

    dataset = create_training_dataset(features)

    assert len(dataset) == 3
    assert [example.target for example in dataset] == [0, 1, 0]
