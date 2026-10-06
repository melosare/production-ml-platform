# Event Data Contract

## Purpose

The event data contract defines the structure and validation requirements for
raw user interaction events entering the ML platform.

The contract is shared by ingestion, validation, feature engineering, and
downstream ML workflows.

## Event Schema

Each event contains the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | string | Yes | Unique identifier for the user |
| `event_timestamp` | datetime | Yes | UTC timestamp when the event occurred |
| `event_type` | string | Yes | Type of user interaction |
| `product_id` | string | Yes | Product associated with the event |
| `session_id` | string | Yes | Identifier for the user session |
| `device_type` | string | Yes | Device used for the interaction |
| `country` | string | Yes | ISO-style country code |
| `value` | float | Yes | Monetary or event value |