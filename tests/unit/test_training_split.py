import pytest

from ml_platform.training.dataset import TrainingExample
from ml_platform.training.split import split_training_dataset


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for split tests."""
    return [
        TrainingExample(
            user_id=f"user_{index:03d}",
            event_count=index,
            session_count=1,
            page_view_count=index,
            feature_usage_count=index // 2,
            target=index % 2,
        )
        for index in range(1, 11)
    ]


def test_split_training_dataset() -> None:
    """The dataset is split into training and test sets."""
    examples = make_examples()

    split = split_training_dataset(examples)

    assert len(split.training) == 8
    assert len(split.test) == 2

    training_ids = {example.user_id for example in split.training}
    test_ids = {example.user_id for example in split.test}

    assert training_ids.isdisjoint(test_ids)
    assert training_ids | test_ids == {example.user_id for example in examples}


def test_split_training_dataset_is_deterministic() -> None:
    """The same inputs produce the same split."""
    examples = make_examples()

    first = split_training_dataset(examples)
    second = split_training_dataset(examples)

    assert first == second


def test_split_training_dataset_preserves_target_distribution() -> None:
    """The split contains both target classes."""
    examples = make_examples()

    split = split_training_dataset(examples)

    assert {example.target for example in split.training} == {0, 1}
    assert {example.target for example in split.test} == {0, 1}


def test_split_training_dataset_rejects_empty_dataset() -> None:
    """An empty dataset is rejected."""
    with pytest.raises(ValueError, match="examples must not be empty"):
        split_training_dataset([])


def test_split_training_dataset_rejects_single_example() -> None:
    """A dataset with fewer than two examples is rejected."""
    example = make_examples()[0]

    with pytest.raises(
        ValueError,
        match="at least two examples are required",
    ):
        split_training_dataset([example])


def test_split_training_dataset_rejects_invalid_test_size() -> None:
    """An invalid test size is rejected."""
    examples = make_examples()

    with pytest.raises(
        ValueError,
        match="test_size must be between 0 and 1",
    ):
        split_training_dataset(examples, test_size=1.0)
