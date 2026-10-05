from pathlib import Path

import yaml

from ml_platform.config.models import ApplicationSettings


def load_settings(path: str | Path) -> ApplicationSettings:
    """App settings from a YAML"""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    return ApplicationSettings(
        environment=configuration["environment"],
        log_level=configuration["log_level"],
    )
