from sklearn.linear_model import LogisticRegression

from ml_platform.inference.predictor import PredictionResult, predict
from ml_platform.model.baseline import ModelInput


class PredictionService:
    """Service for generating model predictions."""

    def __init__(self, model: LogisticRegression) -> None:
        self._model = model

    def predict(self, model_input: ModelInput) -> PredictionResult:
        """Generate a prediction from model input."""
        return predict(self._model, model_input)
