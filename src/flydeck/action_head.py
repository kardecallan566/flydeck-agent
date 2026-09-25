from __future__ import annotations

from dataclasses import dataclass
import math

from .bnb_prediction import Prediction


@dataclass(frozen=True, slots=True)
class ActionProbabilities:
    wait: float
    up: float
    down: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.wait, self.up, self.down)

    @property
    def confidence(self) -> float:
        return max(self.as_tuple())

    @property
    def directional_margin(self) -> float:
        return abs(self.up - self.down)


class ActionProbabilityHead:
    """Tiny non-neural readout separated from the MaleCNS connectome."""

    def __init__(self, learning_rate: float = 0.002, initial_temperature: float = 2.0) -> None:
        if learning_rate <= 0.0 or initial_temperature <= 0.0:
            raise ValueError("invalid action-head parameters")
        self.learning_rate = learning_rate
        self._initial_temperature = initial_temperature
        self._log_temperature = math.log(initial_temperature)
        self.learning_enabled = True
        self._last_logits: tuple[float, float, float] | None = None
        self._last_probabilities: ActionProbabilities | None = None

    @property
    def temperature(self) -> float:
        return math.exp(self._log_temperature)

    @property
    def last_probabilities(self) -> ActionProbabilities | None:
        return self._last_probabilities

    def reset(self, preserve_learning: bool = False) -> None:
        if not preserve_learning:
            self._log_temperature = math.log(self._initial_temperature)
        self._last_logits = None
        self._last_probabilities = None

    def set_learning(self, enabled: bool) -> None:
        self.learning_enabled = enabled

    def predict(self, up_score: float, down_score: float, wait_score: float,
                uncertainty: float = 0.0, novelty: float = 0.0) -> ActionProbabilities:
        # The head receives compact evidence, not the 30K-neuron state.
        logits = (
            wait_score + 0.20 * uncertainty + 0.10 * novelty,
            up_score,
            down_score,
        )
        self._last_logits = logits
        temperature = self.temperature * 0.35
        scaled = [value / max(1e-6, temperature) for value in logits]
        peak = max(scaled)
        exps = [math.exp(min(20.0, value - peak)) for value in scaled]
        total = sum(exps) or 1.0
        probs = ActionProbabilities(exps[0] / total, exps[1] / total, exps[2] / total)
        self._last_probabilities = probs
        return probs

    def observe(self, outcome: Prediction) -> None:
        if not self.learning_enabled or self._last_logits is None:
            return
        target = int(outcome)
        temperature = self.temperature * 0.35
        scaled = [value / max(1e-6, temperature) for value in self._last_logits]
        peak = max(scaled)
        exps = [math.exp(min(20.0, value - peak)) for value in scaled]
        total = sum(exps) or 1.0
        probs = [value / total for value in exps]
        expected = sum(prob * logit for prob, logit in zip(probs, self._last_logits))
        gradient = expected - self._last_logits[target]
        self._log_temperature = max(-math.log(2.0), min(math.log(8.0), self._log_temperature - self.learning_rate * gradient))
