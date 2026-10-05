from ml_platform.health import health_check


def test_health_check() -> None:
    assert health_check() == "healthy"
