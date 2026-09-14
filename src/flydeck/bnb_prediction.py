from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Prediction(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2


@dataclass(frozen=True, slots=True)
class BNBPredictionRound:
    """One 5-minute directional prediction round.

    ``reference_price`` is the price available when the round starts.
    ``close_price`` is only populated when the round has resolved.
    """

    timestamp: int
    reference_price: float
    close_price: float | None = None

    @property
    def resolved(self) -> bool:
        return self.close_price is not None

    @property
    def outcome(self) -> Prediction | None:
        if self.close_price is None:
            return None
        if self.close_price > self.reference_price:
            return Prediction.UP
        if self.close_price < self.reference_price:
            return Prediction.DOWN
        return Prediction.WAIT


@dataclass(frozen=True, slots=True)
class PredictionDecision:
    prediction: Prediction
    confidence: float
    entered: bool


@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction: Prediction
    outcome: Prediction
    correct: bool


class BNBPredictionEnvironment:
    """Minimal environment for the fly's 5-minute BNB behavior.

    This class intentionally contains no portfolio, order, fee, or trade state.
    It exists only to present observations and resolve directional predictions.
    """

    def __init__(self, rounds: tuple[BNBPredictionRound, ...]) -> None:
        if not rounds:
            raise ValueError("rounds must not be empty")
        self.rounds = rounds
        self.index = 0

    @property
    def current(self) -> BNBPredictionRound:
        return self.rounds[self.index]

    def observe(self) -> tuple[float, ...]:
        """Return information available at round start.

        The environment deliberately exposes only current/past information.
        Historical feature engineering belongs outside the outcome resolver so
        the future close can never leak into the observation.
        """
        current = self.current
        return (current.reference_price,)

    def resolve(self, prediction: Prediction) -> PredictionResult:
        current = self.current
        if not current.resolved:
            raise RuntimeError("current round has not resolved")
        outcome = current.outcome
        assert outcome is not None
        return PredictionResult(
            prediction=prediction,
            outcome=outcome,
            correct=prediction in (Prediction.UP, Prediction.DOWN)
            and prediction == outcome,
        )

    def advance(self) -> bool:
        if self.index + 1 >= len(self.rounds):
            return False
        self.index += 1
        return True


class PredictionPolicy:
    """Convert two directional scores into UP/DOWN/WAIT behavior."""

    def __init__(self, confidence_threshold: float = 0.10) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold

    def decide(self, up_score: float, down_score: float) -> PredictionDecision:
        total = abs(up_score) + abs(down_score)
        if total <= 1e-12:
            return PredictionDecision(Prediction.WAIT, 0.0, False)
        if up_score >= down_score:
            confidence = max(0.0, min(1.0, (up_score - down_score) / total))
            prediction = Prediction.UP
        else:
            confidence = max(0.0, min(1.0, (down_score - up_score) / total))
            prediction = Prediction.DOWN
        return PredictionDecision(
            prediction=prediction if confidence >= self.confidence_threshold else Prediction.WAIT,
            confidence=confidence,
            entered=confidence >= self.confidence_threshold,
        )
