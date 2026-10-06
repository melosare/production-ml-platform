import os
from pathlib import Path

from fastapi import FastAPI

from ml_platform.api.app import create_app_from_config


def create_application(config_path: str | Path | None = None) -> FastAPI:
    """Create the application from configuration."""
    if config_path is None:
        config_path = os.getenv(
            "ML_PLATFORM_CONFIG",
            "configs/development.yaml",
        )
    return create_app_from_config(config_path)
