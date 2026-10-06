from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


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
