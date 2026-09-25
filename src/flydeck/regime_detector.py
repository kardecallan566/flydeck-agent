from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from .causal_features import CausalFeatureVector
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
    probabilities: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    duration: int = 0
    previous_regime: MarketRegime = MarketRegime.RANGE


class CausalRegimeDetector:
    """Online prototype detector with causal probabilities and hysteresis."""

    def __init__(self, trend_alpha: float = 0.12, volatility_alpha: float = 0.10,
                 trend_threshold: float = 0.045, persistence_threshold: float = 0.25,
                 enter_threshold: float = 0.55, exit_threshold: float = 0.38) -> None:
        if not 0.0 < trend_alpha <= 1.0 or not 0.0 < volatility_alpha <= 1.0:
            raise ValueError("regime EMA alphas must be in (0, 1]")
        if not 0.0 < exit_threshold < enter_threshold <= 1.0:
            raise ValueError("invalid regime hysteresis thresholds")
        self.trend_alpha = trend_alpha
        self.volatility_alpha = volatility_alpha
        self.trend_threshold = trend_threshold
        self.persistence_threshold = persistence_threshold
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
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

    def step(self, stimulus: RetinaStimulus, features: CausalFeatureVector | None = None) -> RegimeState:
        if features is None:
            velocity = max(-1.0, min(1.0, stimulus.velocity))
            persistence = max(0.0, min(1.0, stimulus.coherence * 0.65 + stimulus.short_coherence * 0.35))
            volatility = max(0.0, min(1.0, stimulus.volatility_contrast / 2.0))
            acceleration = stimulus.acceleration
            volume_signal = max(0.0, stimulus.volume_contrast - 1.0) / 1.5
        else:
            velocity = max(-1.0, min(1.0, 0.55 * features.short_return + 0.30 * features.medium_return + 0.15 * features.long_return))
            persistence = max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(features.values[-4] if len(features.values) >= 4 else 0.0)))
            volatility = max(0.0, min(1.0, features.volatility))
            acceleration = features.values[-3] if len(features.values) >= 3 else 0.0
            volume_signal = max(0.0, features.volume_relative)
        self._trend_ema = (1.0 - self.trend_alpha) * self._trend_ema + self.trend_alpha * velocity
        self._volatility_ema = (1.0 - self.volatility_alpha) * self._volatility_ema + self.volatility_alpha * volatility
        acceleration_jump = abs(velocity - self._previous_velocity)
        self._previous_velocity = velocity
        shock_score = max(
            min(1.0, abs(acceleration)) * 0.45,
            min(1.0, abs(velocity)) * 0.25 + self._volatility_ema * 0.30 + min(1.0, volume_signal) * 0.25,
            min(1.0, acceleration_jump) * 0.55 + self._volatility_ema * 0.25,
        )
        trend_score = max(-1.0, min(1.0, self._trend_ema * (0.55 + 0.45 * persistence)))
        trend_strength = abs(trend_score) * (0.70 + 0.30 * persistence)
        range_strength = max(0.0, 1.0 - abs(trend_score)) * max(0.05, 1.0 - 0.70 * persistence)
        raw = (
            max(0.0, trend_score) * (0.70 + 0.30 * persistence),
            max(0.0, -trend_score) * (0.70 + 0.30 * persistence),
            range_strength,
            shock_score,
        )
        total = sum(math.exp(min(8.0, value * 4.0)) for value in raw)
        probabilities = tuple(math.exp(min(8.0, value * 4.0)) / total for value in raw)
        candidates = (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.RANGE, MarketRegime.SHOCK)
        best_index = max(range(4), key=lambda i: probabilities[i])
        current_index = candidates.index(self._state.regime)
        current_probability = probabilities[current_index]
        if probabilities[best_index] >= self.enter_threshold or current_probability < self.exit_threshold:
            regime = candidates[best_index]
        else:
            regime = self._state.regime
        if regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN) and abs(trend_score) < self.trend_threshold:
            regime = MarketRegime.RANGE if current_probability < self.enter_threshold else regime
        duration = self._state.duration + 1 if regime == self._state.regime else 1
        self._state = RegimeState(regime, trend_score, self._volatility_ema, shock_score, persistence,
                                  probabilities, duration, self._state.regime)
        return self._state
