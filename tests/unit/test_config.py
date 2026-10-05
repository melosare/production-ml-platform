from pathlib import Path

import pytest

from ml_platform.config.settings import load_settings


def test_load_development_settings() -> None:
    """Dev loads correctly."""
    config_path = Path("configs/development.yaml")

    settings = load_settings(config_path)

    assert settings.environment == "development"
    assert settings.log_level == "DEBUG"


def test_missing_configuration_file() -> None:
    """Missing: raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_settings("configs/does-not-exist.yaml")


def test_load_production_settings() -> None:
    """Prod loads correctly."""
    config_path = Path("configs/production.yaml")

    settings = load_settings(config_path)

    assert settings.environment == "production"
    assert settings.log_level == "INFO"
