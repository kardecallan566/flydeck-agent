from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .market_retina import RetinaStimulus


class MarketRegime(StrEnum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    SHOCK = "SHOCK"


@dataclass(frozen=True, slots=True)
class RegimeState:
    regime: MarketRegime
    trend_score: float
    volatility_score: float
    shock_score: float
    persistence: float


class CausalRegimeDetector:
    """Online regime classifier using only the current and prior retina state."""

    def __init__(self, trend_alpha: float = 0.12, volatility_alpha: float = 0.10) -> None:
        if not 0.0 < trend_alpha <= 1.0 or not 0.0 < volatility_alpha <= 1.0:
            raise ValueError("regime EMA alphas must be in (0, 1]")
        self.trend_alpha = trend_alpha
        self.volatility_alpha = volatility_alpha
        self._trend_ema = 0.0
        self._volatility_ema = 0.0
        self._previous_velocity = 0.0
        self._state = RegimeState(MarketRegime.RANGE, 0.0, 0.0, 0.0, 0.0)

    @property
    def state(self) -> RegimeState:
        return self._state

    def reset(self) -> None:
        self._trend_ema = 0.0
        self._volatility_ema = 0.0
        self._previous_velocity = 0.0
        self._state = RegimeState(MarketRegime.RANGE, 0.0, 0.0, 0.0, 0.0)

    def step(self, stimulus: RetinaStimulus) -> RegimeState:
        velocity = max(-1.0, min(1.0, stimulus.velocity))
        self._trend_ema = (1.0 - self.trend_alpha) * self._trend_ema + self.trend_alpha * velocity
        raw_volatility = max(0.0, min(1.0, stimulus.volatility_contrast / 2.0))
        self._volatility_ema = (1.0 - self.volatility_alpha) * self._volatility_ema + self.volatility_alpha * raw_volatility
        acceleration_jump = abs(velocity - self._previous_velocity)
        self._previous_velocity = velocity

        shock_score = max(
            abs(stimulus.acceleration) * 0.45,
            min(1.0, abs(velocity)) * 0.25
            + self._volatility_ema * 0.20
            + max(0.0, stimulus.volume_contrast - 1.0) / 1.5 * 0.30,
            min(1.0, acceleration_jump) * 0.55 + self._volatility_ema * 0.25,
        )
        persistence = max(0.0, min(1.0, stimulus.coherence * 0.65 + stimulus.short_coherence * 0.35))
        trend_score = max(-1.0, min(1.0, self._trend_ema * (0.55 + 0.45 * persistence)))

        if shock_score >= 0.72:
            regime = MarketRegime.SHOCK
        elif trend_score >= 0.16 and persistence >= 0.42:
            regime = MarketRegime.TREND_UP
        elif trend_score <= -0.16 and persistence >= 0.42:
            regime = MarketRegime.TREND_DOWN
        else:
            regime = MarketRegime.RANGE
        self._state = RegimeState(regime, trend_score, self._volatility_ema, shock_score, persistence)
        return self._state
