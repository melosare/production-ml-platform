from pathlib import Path

import pytest

from ml_platform.config.settings import load_settings


def test_load_development_settings() -> None:
    """Development settings load correctly."""
    config_path = Path("configs/development.yaml")

    settings = load_settings(config_path)

    assert settings.environment == "development"
    assert settings.log_level == "DEBUG"
    assert settings.model_path == "artifacts/baseline_model.joblib"


def test_missing_configuration_file() -> None:
    """Missing configuration raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_settings("configs/does-not-exist.yaml")


def test_load_production_settings() -> None:
    """Production settings load correctly."""
    config_path = Path("configs/production.yaml")

    settings = load_settings(config_path)

    assert settings.environment == "production"
    assert settings.log_level == "INFO"
    assert settings.model_path == "artifacts/baseline_model.joblib"
