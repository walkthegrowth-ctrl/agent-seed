"""Small number-generator tool (self-contained; a `tools/` reservoir file)."""

from __future__ import annotations

import random as _random


def random_int(low: int, high: int) -> int:
    """A random integer in [low, high] (inclusive).

    Args:
        low: lower bound (inclusive).
        high: upper bound (inclusive).
    """
    return _random.randint(low, high)
