from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicationSettings:
    """App-level settings."""

    environment: str
    log_level: str
