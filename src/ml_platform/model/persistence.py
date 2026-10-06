import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a persisted model artifact."""

    model_version: str
    model_type: str
    feature_names: list[str]
    random_state: int
    training_examples: int
    validation_examples: int
    test_examples: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


def save_model(model: LogisticRegression, path: str | Path) -> None:
    """Persist a trained logistic regression model to disk."""
    model_path = Path(path)

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)


def load_model(path: str | Path) -> LogisticRegression:
    """Load a persisted logistic regression model from disk."""
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    model = joblib.load(model_path)

    if not isinstance(model, LogisticRegression):
        raise TypeError("persisted object is not a LogisticRegression")

    return model


def save_model_metadata(
    metadata: ModelMetadata,
    path: str | Path,
) -> None:
    """Persist model metadata as JSON."""
    metadata_path = Path(path)

    if not metadata_path.parent.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_metadata(path: str | Path) -> ModelMetadata:
    """Load persisted model metadata from JSON."""
    metadata_path = Path(path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"model metadata file does not exist: {metadata_path}",
        )

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    return ModelMetadata(**data)
