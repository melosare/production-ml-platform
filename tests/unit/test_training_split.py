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
        for index in range(1, 21)
    ]


def test_split_training_dataset() -> None:
    """The dataset is split into training, validation, and test sets."""
    examples = make_examples()

    split = split_training_dataset(examples)

    assert len(split.training) == 14
    assert len(split.validation) == 2
    assert len(split.test) == 4


def test_split_training_dataset_partitions_are_disjoint() -> None:
    """The three partitions do not contain overlapping examples."""
    examples = make_examples()

    split = split_training_dataset(examples)

    training_ids = {example.user_id for example in split.training}
    validation_ids = {example.user_id for example in split.validation}
    test_ids = {example.user_id for example in split.test}

    assert training_ids.isdisjoint(validation_ids)
    assert training_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_split_training_dataset_is_exhaustive() -> None:
    """Every input example appears in exactly one partition."""
    examples = make_examples()

    split = split_training_dataset(examples)

    input_ids = {example.user_id for example in examples}
    split_ids = (
        {example.user_id for example in split.training}
        | {example.user_id for example in split.validation}
        | {example.user_id for example in split.test}
    )

    assert split_ids == input_ids


def test_split_training_dataset_is_deterministic() -> None:
    """The same inputs produce the same split."""
    examples = make_examples()

    first = split_training_dataset(examples)
    second = split_training_dataset(examples)

    assert first == second


def test_split_training_dataset_preserves_target_distribution() -> None:
    """Every partition contains both target classes."""
    examples = make_examples()

    split = split_training_dataset(examples)

    assert {example.target for example in split.training} == {0, 1}
    assert {example.target for example in split.validation} == {0, 1}
    assert {example.target for example in split.test} == {0, 1}


def test_split_training_dataset_accepts_custom_sizes() -> None:
    """Custom validation and test sizes are respected."""
    examples = make_examples()

    split = split_training_dataset(
        examples,
        validation_size=0.2,
        test_size=0.2,
    )

    assert len(split.training) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4


def test_split_training_dataset_rejects_empty_dataset() -> None:
    """An empty dataset is rejected."""
    with pytest.raises(ValueError, match="examples must not be empty"):
        split_training_dataset([])


def test_split_training_dataset_rejects_too_few_examples() -> None:
    """A dataset with fewer than three examples is rejected."""
    example = make_examples()[0]

    with pytest.raises(
        ValueError,
        match="at least three examples are required",
    ):
        split_training_dataset([example, example])


@pytest.mark.parametrize(
    "validation_size",
    [0.0, 1.0, -0.1],
)
def test_split_training_dataset_rejects_invalid_validation_size(
    validation_size: float,
) -> None:
    """An invalid validation size is rejected."""
    examples = make_examples()

    with pytest.raises(
        ValueError,
        match="validation_size must be between 0 and 1",
    ):
        split_training_dataset(
            examples,
            validation_size=validation_size,
        )


@pytest.mark.parametrize(
    "test_size",
    [0.0, 1.0, -0.1],
)
def test_split_training_dataset_rejects_invalid_test_size(
    test_size: float,
) -> None:
    """An invalid test size is rejected."""
    examples = make_examples()

    with pytest.raises(
        ValueError,
        match="test_size must be between 0 and 1",
    ):
        split_training_dataset(
            examples,
            test_size=test_size,
        )


def test_split_training_dataset_rejects_sizes_that_leave_no_training_data() -> None:
    """Validation and test sizes must leave data for training."""
    examples = make_examples()

    with pytest.raises(
        ValueError,
        match="validation_size \\+ test_size must be less than 1",
    ):
        split_training_dataset(
            examples,
            validation_size=0.5,
            test_size=0.5,
        )
