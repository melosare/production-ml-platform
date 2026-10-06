from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from ml_platform.training.dataset import TrainingExample


@dataclass(frozen=True)
class DatasetSplit:
    """Three-way train, validation, and test dataset split."""

    training: list[TrainingExample]
    validation: list[TrainingExample]
    test: list[TrainingExample]


def split_training_dataset(
    examples: list[TrainingExample],
    validation_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split examples into deterministic, stratified train/validation/test sets."""
    if not examples:
        raise ValueError("examples must not be empty")

    if len(examples) < 3:
        raise ValueError("at least three examples are required")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if validation_size + test_size >= 1:
        raise ValueError(
            "validation_size + test_size must be less than 1",
        )

    targets = [example.target for example in examples]

    training_and_validation, test = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    training_and_validation_targets = [
        example.target for example in training_and_validation
    ]

    validation_fraction = validation_size / (1 - test_size)

    training, validation = train_test_split(
        training_and_validation,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=training_and_validation_targets,
    )

    return DatasetSplit(
        training=training,
        validation=validation,
        test=test,
    )
