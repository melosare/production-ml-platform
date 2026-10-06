from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from ml_platform.training.dataset import TrainingExample


@dataclass(frozen=True)
class DatasetSplit:
    """Deterministic training and test split."""

    training: list[TrainingExample]
    test: list[TrainingExample]


def split_training_dataset(
    examples: list[TrainingExample],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split training examples into deterministic training and test sets."""
    if not examples:
        raise ValueError("examples must not be empty")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if len(examples) < 2:
        raise ValueError("at least two examples are required")

    targets = [example.target for example in examples]

    training, test = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    return DatasetSplit(
        training=training,
        test=test,
    )
