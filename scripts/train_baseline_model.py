from pathlib import Path

from ml_platform.training.pipeline import train_and_persist_model

FIXTURE_PATH = Path("tests/fixtures/synthetic_events.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")
METADATA_PATH = Path("artifacts/baseline_model.metadata.json")


def main() -> None:
    """Train and persist the baseline model."""
    metadata = train_and_persist_model(
        data_path=FIXTURE_PATH,
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    print(f"Saved baseline model to {MODEL_PATH}")
    print(f"Saved model metadata to {METADATA_PATH}")
    print(f"Validation metrics: {metadata.validation_metrics}")
    print(f"Test metrics: {metadata.test_metrics}")


if __name__ == "__main__":
    main()
