from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .bnb_prediction import Prediction
from .malecns import MaleCNSCircuit, MaleCNSReservoir


class FlyDecision(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2


@dataclass(frozen=True, slots=True)
class BNBObservation:
    timestamp: int
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
    """Causal BNB sensory system. It never reads future candles."""

    WINDOWS = (1, 2, 3, 6, 12, 24)

    def encode(self, observation: BNBObservation) -> tuple[float, ...]:
        prices = observation.prices
        volumes = observation.volumes
        if not prices or prices[-1] <= 0:
            raise ValueError("prices must contain a positive current price")

        def ret(window: int) -> float:
            if len(prices) <= window:
                return 0.0
            return prices[-1] / prices[-1 - window] - 1.0

        returns = tuple(ret(w) for w in self.WINDOWS)
        short_vol = self._volatility(returns[:3])
        long_vol = self._volatility(returns)
        volume_ratio = 0.0
        if len(volumes) >= 6:
            baseline = sum(volumes[-6:-1]) / 5.0
            if baseline > 0:
                volume_ratio = volumes[-1] / baseline - 1.0
        acceleration = returns[0] - returns[1] / 2.0
        trend = (returns[2] + returns[3] + returns[4]) / 3.0
        recent = prices[-min(len(prices), 12) :]
        lo, hi = min(recent), max(recent)
        range_position = 0.0 if hi <= lo else ((prices[-1] - lo) / (hi - lo) * 2.0 - 1.0)
        raw = returns + (short_vol, long_vol, volume_ratio, acceleration, trend, range_position)
        return tuple(max(-1.0, min(1.0, value * 10.0)) for value in raw)

    @staticmethod
    def _volatility(values: tuple[float, ...]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


class FlyBNBPredictionAgent:
    """MaleCNS behavioral agent whose only behaviors are UP, DOWN and WAIT."""

    def __init__(self, circuit: MaleCNSCircuit, seed: int = 123,
                 confidence_threshold: float = 0.20, learning_rate: float = 0.01) -> None:
        if len(circuit.input_neurons) != 12:
            raise ValueError("BNB sensor requires 12 input pools")
        if len(circuit.output_neurons) != 3:
            raise ValueError("BNB prediction requires three output pools")
        self.sensor = BNBFeatureSensor()
        self.reservoir = MaleCNSReservoir(circuit, feature_count=12, action_count=3, seed=seed)
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        self._last_prediction: FlyPrediction | None = None
        self._last_activities = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.reservoir.reset()
        self._last_prediction = None
        self._last_activities = (0.0, 0.0, 0.0)

    def predict(self, observation: BNBObservation) -> FlyPrediction:
        self.reservoir.step(self.sensor.encode(observation))
        up = self.reservoir.action_activity(1)
        down = self.reservoir.action_activity(2)
        total = abs(up) + abs(down)
        confidence = 0.0 if total <= 1e-12 else abs(up - down) / total
        if confidence < self.confidence_threshold:
            decision = FlyDecision.WAIT
        elif up > down:
            decision = FlyDecision.UP
        else:
            decision = FlyDecision.DOWN
        prediction = FlyPrediction(decision, confidence, up, down)
        self._last_prediction = prediction
        self._last_activities = (
            self.reservoir.action_activity(0), up, down
        )
        return prediction

    def learn(self, outcome: Prediction) -> float:
        prediction = self._last_prediction
        if prediction is None or prediction.decision == FlyDecision.WAIT:
            return 0.0
        correct = (
            prediction.decision == FlyDecision.UP and outcome == Prediction.UP
        ) or (
            prediction.decision == FlyDecision.DOWN and outcome == Prediction.DOWN
        )
        target = 1.0 if correct else -1.0
        action = 1 if prediction.decision == FlyDecision.UP else 2
        error = target - prediction.confidence
        self.reservoir.update_readout(
            action, error, learning_rate=self.learning_rate,
            activity=self._last_activities[action],
        )
        return error


def load_agent(circuit: MaleCNSCircuit, seed: int = 123) -> FlyBNBPredictionAgent:
    return FlyBNBPredictionAgent(circuit, seed=seed)
