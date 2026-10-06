import csv
from datetime import datetime
from pathlib import Path

from ml_platform.data.schemas import Event
from ml_platform.data.validator import validate_event


def load_events(path: str | Path) -> list[Event]:
    """Load and validate events from a CSV file."""
    csv_path = Path(path)

    events: list[Event] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            try:
                event = Event(
                    user_id=row["user_id"],
                    event_timestamp=datetime.fromisoformat(
                        row["event_timestamp"].replace("Z", "+00:00")
                    ),
                    event_type=row["event_type"],
                    product_id=row["product_id"],
                    session_id=row["session_id"],
                    device_type=row["device_type"],
                    country=row["country"],
                    value=float(row["value"]),
                )

                validate_event(event)

            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid event at CSV row {row_number}: {error}"
                ) from error

            events.append(event)

    return events
