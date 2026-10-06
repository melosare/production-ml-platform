from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.features.pipeline import build_features
from ml_platform.model.baseline import train_baseline_model
from ml_platform.model.persistence import save_model
from ml_platform.training.dataset import create_training_dataset

FIXTURE_PATH = Path("tests/fixtures/synthetic_events.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")


def main() -> None:
    """Train and persist the baseline model."""
    events = load_events(FIXTURE_PATH)
    features = build_features(events)
    examples = create_training_dataset(features)

    model = train_baseline_model(examples)
    save_model(model, MODEL_PATH)

    print(f"Saved baseline model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
