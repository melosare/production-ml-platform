import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from ml_platform.config.settings import load_settings
from ml_platform.model.baseline import ModelInput
from ml_platform.model.persistence import load_model
from ml_platform.service.prediction import PredictionService

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request payload for model prediction."""

    event_count: int = Field(ge=0)
    session_count: int = Field(ge=0)
    page_view_count: int = Field(ge=0)
    feature_usage_count: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Response payload for model prediction."""

    predicted_class: int
    probability: float


def create_app(service: PredictionService) -> FastAPI:
    """Create the prediction API application."""
    app = FastAPI(title="Production ML Platform")

    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log request status and duration."""
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "request failed method=%s path=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "request completed method=%s path=%s status_code=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return API liveness status."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """Return API readiness status."""
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Generate a prediction from request data."""
        model_input = ModelInput(
            event_count=request.event_count,
            session_count=request.session_count,
            page_view_count=request.page_view_count,
            feature_usage_count=request.feature_usage_count,
        )

        try:
            result = service.predict(model_input)
        except Exception as exc:
            logger.exception("prediction failed")
            raise HTTPException(
                status_code=500,
                detail="prediction failed",
            ) from exc

        logger.info(
            "prediction completed predicted_class=%s probability=%.4f",
            result.predicted_class,
            result.probability,
        )

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

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info(
        "starting application environment=%s model_path=%s",
        settings.environment,
        settings.model_path,
    )

    return create_app_from_model_path(settings.model_path)
