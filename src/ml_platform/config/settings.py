import os
from pathlib import Path

import yaml

from ml_platform.config.models import ApplicationSettings


def load_settings(path: str | Path) -> ApplicationSettings:
    """Load application settings from YAML with environment overrides."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    return ApplicationSettings(
        environment=os.getenv(
            "ML_PLATFORM_ENVIRONMENT",
            configuration["environment"],
        ),
        log_level=os.getenv(
            "ML_PLATFORM_LOG_LEVEL",
            configuration["log_level"],
        ),
        model_path=os.getenv(
            "ML_PLATFORM_MODEL_PATH",
            configuration["model_path"],
        ),
    )
