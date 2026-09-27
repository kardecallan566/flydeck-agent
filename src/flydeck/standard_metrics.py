from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class StandardizedMetrics:
    """Common definitions used by classification and economic benchmarks."""

    rounds: int
    entered: int
    correct: int
    profitable: int
    accuracy: float
    hit_rate: float
    coverage: float
    economic_return: float


def standardize_metrics(*, rounds: int, entered: int, correct: int, profitable: int,
                        net_returns: list[float] | tuple[float, ...] = ()) -> StandardizedMetrics:
    rounds = max(0, rounds)
    entered = max(0, entered)
    correct = max(0, correct)
    profitable = max(0, profitable)
    equity = 1.0
    for value in net_returns:
        equity *= 1.0 + float(value)
    return StandardizedMetrics(
        rounds=rounds,
        entered=entered,
        correct=correct,
        profitable=profitable,
        accuracy=correct / entered if entered else 0.0,
        hit_rate=profitable / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        economic_return=equity - 1.0,
    )


def direction_is_correct(direction: int, outcome: object) -> bool:
    """Compare +1/-1 with an UP/DOWN-like enum without importing policy code."""
    name = getattr(outcome, "name", str(outcome)).upper()
    return (direction > 0 and name.endswith("UP")) or (direction < 0 and name.endswith("DOWN"))


def is_profitable(net_return: float, tolerance: float = 0.0) -> bool:
    return math.isfinite(net_return) and net_return > tolerance
