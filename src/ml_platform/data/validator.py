from datetime import datetime

from ml_platform.data.schemas import Event

ALLOWED_EVENT_TYPES = {
    "session_start",
    "page_view",
    "feature_used",
    "purchase",
    "session_end",
}


def validate_event(event: Event) -> None:
    """Validate a single event against expectation."""
    required_fields = {
        "user_id": event.user_id,
        "event_type": event.event_type,
        "product_id": event.product_id,
        "session_id": event.session_id,
        "device_type": event.device_type,
        "country": event.country,
    }

    for field_name, value in required_fields.items():
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")

    if event.event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event.event_type}")

    if not isinstance(event.event_timestamp, datetime):
        raise TypeError("event_timestamp must be a datetime")

    if event.value < 0:
        raise ValueError("value must be greater than or equal to 0")
