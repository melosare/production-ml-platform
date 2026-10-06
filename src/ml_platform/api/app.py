from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from ml_platform.config.settings import load_settings
from ml_platform.model.baseline import ModelInput
from ml_platform.model.persistence import load_model
from ml_platform.service.prediction import PredictionService


class PredictionRequest(BaseModel):
    """Request payload for model prediction."""

    event_count: int
    session_count: int
    page_view_count: int
    feature_usage_count: int


class PredictionResponse(BaseModel):
    """Response payload for model prediction."""

    predicted_class: int
    probability: float


def create_app(service: PredictionService) -> FastAPI:
    """Create the prediction API application."""
    app = FastAPI(title="Production ML Platform")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return API health status."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Generate a prediction from request data."""
        model_input = ModelInput(
            event_count=request.event_count,
            session_count=request.session_count,
            page_view_count=request.page_view_count,
            feature_usage_count=request.feature_usage_count,
        )

        result = service.predict(model_input)

        return PredictionResponse(
            predicted_class=result.predicted_class,
            probability=result.probability,
        )

    return app


def create_app_from_model_path(model_path: str | Path) -> FastAPI:
    """Create the prediction API using a persisted model."""
    model = load_model(model_path)
    service = PredictionService(model)

    return create_app(service)


def create_app_from_config(config_path: str | Path) -> FastAPI:
    """Create the prediction API from application configuration."""
    settings = load_settings(config_path)

    return create_app_from_model_path(settings.model_path)
