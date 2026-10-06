from pathlib import Path

import pytest

from ml_platform.model.baseline import train_baseline_model
from ml_platform.model.persistence import load_model, save_model
from ml_platform.training.dataset import TrainingExample


def make_examples() -> list[TrainingExample]:
    """Create deterministic examples for persistence tests."""
    return [
        TrainingExample(
            user_id=f"user_{index:03d}",
            event_count=10 + index,
            session_count=2,
            page_view_count=5 + index,
            feature_usage_count=2,
            target=index % 2,
        )
        for index in range(10)
    ]


def test_save_and_load_model(tmp_path: Path) -> None:
    """A saved model can be loaded with its predictions preserved."""
    examples = make_examples()
    model = train_baseline_model(examples)

    model_path = tmp_path / "baseline_model.joblib"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert loaded_model.classes_.tolist() == model.classes_.tolist()
    assert (
        loaded_model.predict([[12, 2, 7, 2]]).tolist()
        == model.predict([[12, 2, 7, 2]]).tolist()
    )


def test_save_model_creates_parent_directory(tmp_path: Path) -> None:
    """Saving a model creates missing parent directories."""
    examples = make_examples()
    model = train_baseline_model(examples)

    model_path = tmp_path / "models" / "baseline_model.joblib"

    save_model(model, model_path)

    assert model_path.exists()


def test_load_model_rejects_missing_file(tmp_path: Path) -> None:
    """Loading a missing model file raises FileNotFoundError."""
    model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="model file does not exist",
    ):
        load_model(model_path)
