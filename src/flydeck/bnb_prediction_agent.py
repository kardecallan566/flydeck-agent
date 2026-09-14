from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import exp, log
from random import Random

from .bnb_prediction import Prediction
from .malecns import MaleCNSCircuit, MaleCNSReservoir


class FlyDecision(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2


@dataclass(frozen=True, slots=True)
class BNBObservation:
    prices: tuple[float, ...]
    volumes: tuple[float, ...]

    @property
    def current_price(self) -> float:
        return self.prices[-1]


@dataclass(frozen=True, slots=True)
class FlyPrediction:
    decision: FlyDecision
    confidence: float
    up_score: float
    down_score: float


class BNBFeatureSensor:
    """Convert the causal BNB history into a compact sensory signal.

    Only prices/volume up to the current five-minute round are accepted.
    Features are relative rather than absolute, making the sensor independent
    of the nominal BNB price level.
    """

    WINDOWS = (1, 2, 3, 6, 12, 24)

    def encode(self, observation: BNBObservation) -> tuple[float, ...]:
        prices = observation.prices
        volumes = observation.volumes
        current = prices[-1]
        if current <= 0:
            raise ValueError("current price must be positive")

        def ret(window: int) -> float:
            if len(prices) <= window:
                return 0.0
            return prices[-1] / prices[-1 - window] - 1.0

        returns = tuple(ret(w) for w in self.WINDOWS)
        short_vol = self._volatility(returns[:3])
        long_vol = self._volatility(returns)
        volume_ratio = 0.0
        if len(volumes) >= 6:
            base = sum(volumes[-6:-1]) / 5.0
            if base > 0:
                volume_ratio = volumes[-1] / base - 1.0

        acceleration = returns[0] - returns[1] / 2.0
        trend = (returns[2] + returns[3] + returns[4]) / 3.0
        range_position = 0.0
        recent = prices[-min(len(prices), 12) :]
        lo, hi = min(recent), max(recent)
        if hi > lo:
            range_position = (current - lo) / (hi - lo) * 2.0 - 1.0

        raw = returns + (
            short_vol,
            long_vol,
            volume_ratio,
            acceleration,
            trend,
            range_position,
        )
        return tuple(max(-1.0, min(1.0, x * 10.0)) for x in raw)

    @staticmethod
    def _volatility(values: tuple[float, ...]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


class FlyBNBPredictionAgent:
    """MaleCNS-based behavioral agent for five-minute BNB direction."""

    def __init__(
        self,
        circuit: MaleCNSCircuit,
        seed: int = 123,
        confidence_threshold: float = 0.20,
        learning_rate: float = 0.01,
        memory_decay: float = 0.92,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.sensor = BNBFeatureSensor()
        self.reservoir = MaleCNSReservoir(circuit, seed=seed)
        self.rng = Random(seed)
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        self.memory_decay = memory_decay
        self._memory = 0.0
        self._last_activity: tuple[float, ...] | None = None
        self._last_decision = FlyDecision.WAIT

    def observe(self, observation: BNBObservation) -> FlyPrediction:
        features = self.sensor.encode(observation)
        if self._memory:
            features = features + (max(-1.0, min(1.0, self._memory)),)
        activity = self.reservoir.step(features)
        self._last_activity = activity
        up = self.reservoir.action_activity("BUY")
        down = self.reservoir.action_activity("SELL")
        scale = abs(up) + abs(down) + 1e-12
        confidence = abs(up - down) / scale
        if confidence < self.confidence_threshold:
            decision = FlyDecision.WAIT
        elif up > down:
            decision = FlyDecision.UP
        else:
            decision = FlyDecision.DOWN
        self._last_decision = decision
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * (
            1.0 if decision == FlyDecision.UP else -1.0 if decision == FlyDecision.DOWN else 0.0
        )
        return FlyPrediction(decision, confidence, up, down)

    def learn(self, outcome: Prediction, prediction: FlyPrediction) -> float:
        """Apply outcome feedback only to a prediction the fly actually entered."""
        if prediction.decision == FlyDecision.WAIT:
            return 0.0
        target = 1.0 if (
            prediction.decision == FlyDecision.UP and outcome == Prediction.UP
        ) or (
            prediction.decision == FlyDecision.DOWN and outcome == Prediction.DOWN
        ) else -1.0
        action = "BUY" if prediction.decision == FlyDecision.UP else "SELL"
        activity = self.reservoir.action_activity(action)
        error = target - prediction.confidence
        self.reservoir.update_readout(action, error, learning_rate=self.learning_rate)
        return error

    @property
    def decision(self) -> FlyDecision:
        return self._last_decision


def load_agent(circuit: MaleCNSCircuit, seed: int = 123) -> FlyBNBPredictionAgent:
    return FlyBNBPredictionAgent(circuit, seed=seed)
