from __future__ import annotations

from dataclasses import dataclass
import math

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .visual_agent import VisualDecision


@dataclass(frozen=True, slots=True)
class CryptoEventConfig:
    horizons: tuple[int, ...] = (1, 3, 6)
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    volatility_multiplier: float = 0.75
    min_edge_bps: float = 1.0
    min_directional_edge: float = 0.18
    max_position: float = 0.35

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True, slots=True)
class CryptoEventTarget:
    horizon_returns: tuple[float, ...]
    thresholds: tuple[float, ...]
    labels: tuple[Prediction, ...]


@dataclass(frozen=True, slots=True)
class CryptoEventAction:
    position: float
    direction: float
    confidence: float
    horizon: int
    expected_return: float
    risk: float


class CryptoEventLabeler:
    """Causal-at-decision labeler; future values are used only for evaluation."""

    def __init__(self, config: CryptoEventConfig | None = None) -> None:
        self.config = config or CryptoEventConfig()

    def target(self, data: BNBPredictionDataset, index: int) -> CryptoEventTarget:
        if not 0 <= index < data.size - max(self.config.horizons):
            raise IndexError("target requires all configured future horizons")
        volatility = data.volatility(index, window=24)
        thresholds = tuple(self.config.round_trip_cost + self.config.min_edge_bps / 10_000.0
                           + self.config.volatility_multiplier * volatility * math.sqrt(horizon)
                           for horizon in self.config.horizons)
        returns = tuple(data.closes[index + horizon] / data.closes[index] - 1.0 for horizon in self.config.horizons)
        labels = tuple(
            Prediction.UP if value > threshold else Prediction.DOWN if value < -threshold else Prediction.WAIT
            for value, threshold in zip(returns, thresholds)
        )
        return CryptoEventTarget(returns, thresholds, labels)


class CryptoEventPolicy:
    """Continuous policy overlay: MaleCNS evidence becomes a signed position."""

    def __init__(self, config: CryptoEventConfig | None = None) -> None:
        self.config = config or CryptoEventConfig()

    def decide(self, decision: VisualDecision, *, fast_memory: float = 0.0,
               slow_memory: float = 0.0, uncertainty: float = 0.0,
               novelty: float = 0.0, regime: str = "RANGE",
               temporal_up: float = 0.0, temporal_down: float = 0.0,
               temporal_flat: float = 0.0, temporal_attention: float = 0.0) -> CryptoEventAction:
        directional_mass = max(0.0, min(1.0, decision.p_up + decision.p_down))
        direction = max(-1.0, min(1.0, decision.p_up - decision.p_down))
        score_direction = max(-1.0, min(1.0, decision.up_score - decision.down_score))
        temporal_direction = max(-1.0, min(1.0, temporal_up - temporal_down))
        temporal_weight = min(0.30, max(0.0, temporal_attention) * 0.30)
        direction = max(-1.0, min(1.0, (1.0 - temporal_weight) * (0.65 * direction + 0.35 * score_direction)
                         + temporal_weight * temporal_direction))
        # P(WAIT) is not directional confidence. Counting it here previously
        # created large positions while the multiclass head was abstaining.
        confidence = max(0.0, min(1.0, directional_mass * (0.5 + 0.5 * abs(direction))))
        coherence = max(0.0, min(1.0, 1.0 - abs(fast_memory - slow_memory)))
        risk = max(0.0, min(1.0, 0.45 * uncertainty + 0.20 * novelty + 0.15 * (1.0 - coherence)
                            + 0.20 * temporal_flat * temporal_attention))
        if regime == "SHOCK":
            risk = min(1.0, risk + 0.20)
        # Continuous exposure: no binary veto, only a smooth risk/uncertainty shrinkage.
        directional_edge = abs(direction) * confidence
        usable_edge = max(0.0, directional_edge - self.config.min_directional_edge) / (1.0 - self.config.min_directional_edge)
        if decision.wait:
            usable_edge *= 0.50
        position = self.config.max_position * math.tanh(direction * 3.5) * usable_edge * (1.0 - 0.65 * risk)
        if abs(fast_memory) > abs(slow_memory) * 1.25:
            horizon = self.config.horizons[0]
        elif fast_memory * slow_memory > 0.0:
            horizon = self.config.horizons[-1]
        else:
            horizon = self.config.horizons[min(1, len(self.config.horizons) - 1)]
        expected_return = direction * confidence * (1.0 - risk)
        return CryptoEventAction(position=max(-1.0, min(1.0, position)), direction=direction,
                                 confidence=confidence, horizon=horizon,
                                 expected_return=expected_return, risk=risk)
