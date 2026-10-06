from datetime import UTC, datetime
from pathlib import Path

from ml_platform.data.loader import load_events
from ml_platform.data.schemas import Event
from ml_platform.features.aggregations import aggregate_user_features
from ml_platform.features.pipeline import build_features

FIXTURE_PATH = Path("tests/fixtures/events.csv")


def make_event(
    event_type: str,
    session_id: str = "session_001",
    value: float = 0.0,
) -> Event:
    """Create a test event."""
    return Event(
        user_id="user_001",
        event_timestamp=datetime(2026, 1, 5, 9, 15, tzinfo=UTC),
        event_type=event_type,
        product_id="product_001",
        session_id=session_id,
        device_type="mobile",
        country="CA",
        value=value,
    )


def test_aggregate_user_features() -> None:
    """Events are correctly aggregated into user features."""
    events = [
        make_event("session_start"),
        make_event("page_view"),
        make_event("page_view"),
        make_event("feature_used"),
        make_event("purchase", value=49.99),
        make_event("session_end"),
    ]

    features = aggregate_user_features(events)

    assert features.user_id == "user_001"
    assert features.event_count == 6
    assert features.session_count == 1
    assert features.page_view_count == 2
    assert features.feature_usage_count == 1
    assert features.purchase_count == 1
    assert features.total_purchase_value == 49.99


def test_aggregate_user_features_multiple_sessions() -> None:
    """Session count is calculated correctly."""
    events = [
        make_event("session_start", session_id="session_001"),
        make_event("session_end", session_id="session_001"),
        make_event("session_start", session_id="session_002"),
        make_event("session_end", session_id="session_002"),
    ]

    features = aggregate_user_features(events)

    assert features.session_count == 2


def test_aggregate_user_features_rejects_empty_events() -> None:
    """Empty event collections are rejected."""
    try:
        aggregate_user_features([])
    except ValueError as error:
        assert str(error) == "events must not be empty"
    else:
        raise AssertionError("Expected ValueError")


def test_aggregate_user_features_rejects_multiple_users() -> None:
    """Events belonging to multiple users are rejected."""
    first_event = make_event("page_view")
    second_event = Event(
        user_id="user_002",
        event_timestamp=first_event.event_timestamp,
        event_type="page_view",
        product_id="product_001",
        session_id="session_002",
        device_type="desktop",
        country="US",
        value=0.0,
    )

    try:
        aggregate_user_features([first_event, second_event])
    except ValueError as error:
        assert str(error) == "all events must belong to the same user"
    else:
        raise AssertionError("Expected ValueError")


def test_build_features() -> None:
    """The feature pipeline creates one feature record per user."""
    events = load_events(FIXTURE_PATH)

    features = build_features(events)

    assert len(features) == 5

    user_ids = {feature.user_id for feature in features}

    assert user_ids == {
        "user_001",
        "user_002",
        "user_003",
        "user_004",
        "user_005",
    }


def test_build_features_user_001() -> None:
    """The pipeline calculates the expected features for user_001."""
    events = load_events(FIXTURE_PATH)

    features = build_features(events)

    user_features = next(
        feature for feature in features if feature.user_id == "user_001"
    )

    assert user_features.event_count == 5
    assert user_features.session_count == 1
    assert user_features.page_view_count == 1
    assert user_features.feature_usage_count == 1
    assert user_features.purchase_count == 1
    assert user_features.total_purchase_value == 49.99
