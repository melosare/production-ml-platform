import csv
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_events.csv"


def add_event(
    events: list[dict[str, str]],
    user_number: int,
    user_id: str,
    session_id: str,
    base_timestamp: datetime,
    event_type: str,
    timestamp_offset: int,
    value: float = 0.0,
) -> None:
    """Add one deterministic event to the dataset."""
    timestamp = base_timestamp + timedelta(minutes=timestamp_offset)

    events.append(
        {
            "user_id": user_id,
            "event_timestamp": timestamp.isoformat() + "Z",
            "event_type": event_type,
            "product_id": f"product_{(user_number % 10) + 1:02d}",
            "session_id": session_id,
            "device_type": ["mobile", "desktop", "tablet"][user_number % 3],
            "country": ["CA", "US", "GB", "AU"][user_number % 4],
            "value": f"{value:.2f}",
        }
    )


def generate_events() -> list[dict[str, str]]:
    """Generate a deterministic synthetic event dataset."""
    events: list[dict[str, str]] = []

    for user_number in range(1, 101):
        user_id = f"user_{user_number:03d}"

        # Deterministic behavioral groups:
        # 1-40: low activity, no purchase
        # 41-70: moderate activity, mixed behavior
        # 71-100: high activity, purchase
        if user_number <= 40:
            session_count = 1
            page_view_count = 1 + (user_number % 2)
            feature_usage_count = user_number % 2
            purchase = False
        elif user_number <= 70:
            session_count = 2 + (user_number % 2)
            page_view_count = 3 + (user_number % 3)
            feature_usage_count = 1 + (user_number % 2)
            purchase = user_number % 3 != 0
        else:
            session_count = 3 + (user_number % 3)
            page_view_count = 5 + (user_number % 4)
            feature_usage_count = 2 + (user_number % 3)
            purchase = True

        base_timestamp = datetime(2026, 1, 1, 12, 0, 0) + timedelta(days=user_number)

        for session_number in range(1, session_count + 1):
            session_id = f"session_{user_number:03d}_{session_number:02d}"

            session_offset = (session_number - 1) * 60

            add_event(
                events,
                user_number,
                user_id,
                session_id,
                base_timestamp,
                "session_start",
                session_offset,
            )

            # Spread page views across sessions deterministically.
            for page_view_number in range(page_view_count):
                add_event(
                    events,
                    user_number,
                    user_id,
                    session_id,
                    base_timestamp,
                    "page_view",
                    session_offset + 1 + page_view_number,
                )

            # Feature usage occurs only in the first session.
            if session_number == 1:
                for feature_number in range(feature_usage_count):
                    add_event(
                        events,
                        user_number,
                        user_id,
                        session_id,
                        base_timestamp,
                        "feature_used",
                        session_offset + 10 + feature_number,
                    )

                if purchase:
                    purchase_value = 20.0 + (user_number % 10) * 5.0

                    add_event(
                        events,
                        user_number,
                        user_id,
                        session_id,
                        base_timestamp,
                        "purchase",
                        session_offset + 20,
                        purchase_value,
                    )

            add_event(
                events,
                user_number,
                user_id,
                session_id,
                base_timestamp,
                "session_end",
                session_offset + 30,
            )

    return events


def main() -> None:
    """Generate and write the synthetic event dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    events = generate_events()

    fieldnames = [
        "user_id",
        "event_timestamp",
        "event_type",
        "product_id",
        "session_id",
        "device_type",
        "country",
        "value",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    print(f"Generated {len(events)} events for 100 users.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
