from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from ml_platform.model.baseline import ModelInput


@dataclass(frozen=True)
class PredictionResult:
    """Prediction produced by the baseline model."""

    predicted_class: int
    probability: float


def predict(
    model: LogisticRegression,
    model_input: ModelInput,
) -> PredictionResult:
    """Generate a prediction for a single model input."""
    values = [
        [
            model_input.event_count,
            model_input.session_count,
            model_input.page_view_count,
            model_input.feature_usage_count,
        ]
    ]

    predicted_class = int(model.predict(values)[0])
    probability = float(model.predict_proba(values)[0][1])

    return PredictionResult(
        predicted_class=predicted_class,
        probability=probability,
    )
