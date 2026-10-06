from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.evaluation.evaluator import evaluate_model
from ml_platform.features.pipeline import build_features
from ml_platform.model.baseline import train_baseline_model
from ml_platform.model.persistence import (
    ModelMetadata,
    save_model,
    save_model_metadata,
)
from ml_platform.training.dataset import create_training_dataset
from ml_platform.training.split import split_training_dataset

MODEL_VERSION = "baseline-1.0.0"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "event_count",
    "session_count",
    "page_view_count",
    "feature_usage_count",
]


def train_and_persist_model(
    data_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
) -> ModelMetadata:
    """Train, evaluate, and persist the baseline model."""
    events = load_events(data_path)
    features = build_features(events)
    examples = create_training_dataset(features)

    split = split_training_dataset(
        examples,
        random_state=RANDOM_STATE,
    )

    model = train_baseline_model(split.training)

    validation_result = evaluate_model(model, split.validation)
    test_result = evaluate_model(model, split.test)

    metadata = ModelMetadata(
        model_version=MODEL_VERSION,
        model_type="LogisticRegression",
        feature_names=FEATURE_NAMES,
        random_state=RANDOM_STATE,
        training_examples=len(split.training),
        validation_examples=len(split.validation),
        test_examples=len(split.test),
        validation_metrics={
            "accuracy": validation_result.metrics.accuracy,
            "precision": validation_result.metrics.precision,
            "recall": validation_result.metrics.recall,
        },
        test_metrics={
            "accuracy": test_result.metrics.accuracy,
            "precision": test_result.metrics.precision,
            "recall": test_result.metrics.recall,
        },
    )

    save_model(model, model_path)
    save_model_metadata(metadata, metadata_path)

    return metadata
