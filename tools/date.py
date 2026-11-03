"""Current date & time tool (self-contained; a `tools/` reservoir file)."""

from __future__ import annotations

from datetime import UTC, date


def current_date() -> str:
    """Today's date (YYYY-MM-DD) in UTC."""
    return date.today(UTC).strftime("%Y-%m-%d")
