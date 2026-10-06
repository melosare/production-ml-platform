from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Event:
    """dummy user telemetry data"""

    user_id: str
    event_timestamp: datetime
    event_type: str
    product_id: str
    session_id: str
    device_type: str
    country: str
    value: float
