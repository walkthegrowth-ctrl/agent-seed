"""Current date & time tool (self-contained; a `tools/` reservoir file)."""

from __future__ import annotations

from datetime import UTC, datetime


def current_date() -> str:
    """Today's date (YYYY-MM-DD) in UTC."""
    return datetime.now(UTC).date().strftime("%Y-%m-%d")
