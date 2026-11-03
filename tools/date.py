"""Current date & time tool (self-contained; a `tools/` reservoir file)."""

from __future__ import annotations

from datetime import UTC, date


def current_date(now_utc: str = "now") -> str:
    """Today's date (YYYY-MM-DD) in UTC.

    Args:
        now_utc: reserved (always ignored); the function always reports "now".
    """
    return date.today(UTC).strftime("%Y-%m-%d")
