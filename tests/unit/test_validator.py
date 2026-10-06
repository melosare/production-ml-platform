from datetime import UTC, datetime

import pytest

from ml_platform.data.schemas import Event
from ml_platform.data.validator import validate_event


def make_valid_event() -> Event:
    """Create a valid event"""
    return Event(
        user_id="user_001",
        event_timestamp=datetime(2026, 1, 5, 9, 15, tzinfo=UTC),
        event_type="page_view",
        product_id="product_001",
        session_id="session_001",
        device_type="mobile",
        country="CA",
        value=0.0,
    )


def test_valid_event_passes_validation() -> None:
    """Valid events should pass validation."""
    validate_event(make_valid_event())


def test_empty_user_id_fails_validation() -> None:
    """Events with an empty user ID should fail validation."""
    event = make_valid_event()
    invalid_event = Event(
        user_id="",
        event_timestamp=event.event_timestamp,
        event_type=event.event_type,
        product_id=event.product_id,
        session_id=event.session_id,
        device_type=event.device_type,
        country=event.country,
        value=event.value,
    )

    with pytest.raises(ValueError, match="user_id must not be empty"):
        validate_event(invalid_event)


def test_invalid_event_type_fails_validation() -> None:
    """Unsupported event types should fail validation."""
    event = make_valid_event()
    invalid_event = Event(
        user_id=event.user_id,
        event_timestamp=event.event_timestamp,
        event_type="invalid_event",
        product_id=event.product_id,
        session_id=event.session_id,
        device_type=event.device_type,
        country=event.country,
        value=event.value,
    )

    with pytest.raises(ValueError, match="Unsupported event type"):
        validate_event(invalid_event)


def test_negative_value_fails_validation() -> None:
    """Negative event values should fail validation."""
    event = make_valid_event()
    invalid_event = Event(
        user_id=event.user_id,
        event_timestamp=event.event_timestamp,
        event_type=event.event_type,
        product_id=event.product_id,
        session_id=event.session_id,
        device_type=event.device_type,
        country=event.country,
        value=-1.0,
    )

    with pytest.raises(
        ValueError,
        match="value must be greater than or equal to 0",
    ):
        validate_event(invalid_event)
