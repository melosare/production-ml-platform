from dataclasses import dataclass

from ml_platform.data.schemas import Event


@dataclass(frozen=True)
class UserFeatures:
    """Aggregated features."""

    user_id: str
    event_count: int
    session_count: int
    page_view_count: int
    feature_usage_count: int
    purchase_count: int
    total_purchase_value: float


def aggregate_user_features(events: list[Event]) -> UserFeatures:
    """Aggregate events into features."""
    if not events:
        raise ValueError("events must not be empty")

    user_ids = {event.user_id for event in events}

    if len(user_ids) != 1:
        raise ValueError("all events must belong to the same user")

    user_id = events[0].user_id

    session_ids = {event.session_id for event in events}
    page_view_count = sum(event.event_type == "page_view" for event in events)
    feature_usage_count = sum(event.event_type == "feature_used" for event in events)
    purchase_count = sum(event.event_type == "purchase" for event in events)
    total_purchase_value = sum(
        event.value for event in events if event.event_type == "purchase"
    )

    return UserFeatures(
        user_id=user_id,
        event_count=len(events),
        session_count=len(session_ids),
        page_view_count=page_view_count,
        feature_usage_count=feature_usage_count,
        purchase_count=purchase_count,
        total_purchase_value=total_purchase_value,
    )
