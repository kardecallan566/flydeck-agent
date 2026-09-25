from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visual_agent import VisualDecision


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """Compact event signature; contains only information available at decision time."""

    timestamp: int
    state: tuple[float, ...]
    direction: float
    intensity: float
    volatility: float
    volume_surprise: float
    novelty: float
    regime: str


class TemporalEventExtractor:
    """Extract sparse market events from causal prices and MaleCNS readouts."""

    def __init__(self, min_move: float = 0.00015, volume_weight: float = 0.20) -> None:
        self.min_move = max(0.0, min_move)
        self.volume_weight = max(0.0, min(1.0, volume_weight))

    def extract(self, *, timestamp: int, prices: tuple[float, ...], volumes: tuple[float, ...],
                decision: VisualDecision, novelty: float = 0.0,
                regime: str | None = None) -> TemporalEvent:
        if len(prices) < 2:
            raise ValueError("temporal event extraction needs at least two prices")
        returns = tuple(prices[i] / max(1e-12, prices[i - 1]) - 1.0 for i in range(1, len(prices)))
        short = self._sum_return(returns, 1)
        medium = self._sum_return(returns, min(3, len(returns)))
        long = self._sum_return(returns, min(8, len(returns)))
        volatility = self._rms(returns[-min(8, len(returns)):])
        volume_surprise = self._volume_surprise(volumes)
        direction = max(-1.0, min(1.0, 0.45 * math.tanh(short * 600.0)
                         + 0.35 * math.tanh(medium * 250.0)
                         + 0.20 * math.tanh(long * 100.0)))
        intensity = max(0.0, min(1.0, abs(direction) * 0.60
                                + min(1.0, volatility * 400.0) * 0.25
                                + volume_surprise * self.volume_weight))
        state = (
            max(-1.0, min(1.0, decision.p_up - decision.p_down)),
            max(-1.0, min(1.0, decision.up_score - decision.down_score)),
            direction,
            min(1.0, volatility * 400.0),
            volume_surprise,
            max(0.0, min(1.0, decision.p_wait)),
            max(0.0, min(1.0, novelty)),
        )
        return TemporalEvent(timestamp=timestamp, state=state, direction=direction,
                             intensity=intensity, volatility=volatility,
                             volume_surprise=volume_surprise,
                             novelty=max(0.0, min(1.0, novelty)),
                             regime=regime or decision.regime)

    @staticmethod
    def _sum_return(values: tuple[float, ...], count: int) -> float:
        return sum(values[-count:]) if values else 0.0

    @staticmethod
    def _rms(values: tuple[float, ...]) -> float:
        return math.sqrt(sum(value * value for value in values) / max(1, len(values)))

    @staticmethod
    def _volume_surprise(volumes: tuple[float, ...]) -> float:
        if len(volumes) < 4:
            return 0.0
        baseline = sum(volumes[:-1]) / max(1, len(volumes) - 1)
        if baseline <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, (volumes[-1] / baseline - 1.0) / 3.0))
