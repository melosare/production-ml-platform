from pathlib import Path

import pytest

from ml_platform.data.loader import load_events

FIXTURE_PATH = Path("tests/fixtures/events.csv")


def test_load_events() -> None:
    """The event fixture loads into validated Event objects."""
    events = load_events(FIXTURE_PATH)

    assert len(events) == 24

    first_event = events[0]

    assert first_event.user_id == "user_001"
    assert first_event.event_type == "session_start"
    assert first_event.product_id == "product_001"
    assert first_event.session_id == "session_001"
    assert first_event.device_type == "mobile"
    assert first_event.country == "CA"
    assert first_event.value == 0.0


def test_load_events_contains_purchase() -> None:
    """The loader correctly parses purchase events and values."""
    events = load_events(FIXTURE_PATH)

    purchases = [event for event in events if event.event_type == "purchase"]

    assert len(purchases) == 3
    assert purchases[0].value == 49.99


def test_missing_events_file() -> None:
    """A missing event file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_events("tests/fixtures/does-not-exist.csv")
