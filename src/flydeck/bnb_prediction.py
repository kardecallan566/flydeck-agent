from __future__ import annotations

from enum import IntEnum


class Prediction(IntEnum):
    """Resolved direction of the next BNB 5-minute candle."""

    WAIT = 0
    UP = 1
    DOWN = 2
