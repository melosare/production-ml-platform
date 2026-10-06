from collections import defaultdict

from ml_platform.data.schemas import Event
from ml_platform.features.aggregations import (
    UserFeatures,
    aggregate_user_features,
)


def build_features(events: list[Event]) -> list[UserFeatures]:
    """Build aggregated features."""
    events_by_user: dict[str, list[Event]] = defaultdict(list)

    for event in events:
        events_by_user[event.user_id].append(event)

    return [
        aggregate_user_features(user_events) for user_events in events_by_user.values()
    ]
