from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bnb_prediction import Prediction
from .statistical_metrics import BinaryEvaluation, evaluate_binary


@dataclass(frozen=True, slots=True)
class SelectivePolicy:
    """Frozen UP/DOWN/WAIT rule calibrated on validation data only."""

    confidence_threshold: float

    def decide(self, p_up: float) -> Prediction:
        probability = min(1.0, max(0.0, float(p_up)))
        confidence = abs(probability - 0.5) * 2.0
        if confidence < self.confidence_threshold:
            return Prediction.WAIT
        return Prediction.UP if probability >= 0.5 else Prediction.DOWN

    def apply(self, probabilities: Sequence[float]) -> tuple[Prediction, ...]:
        return tuple(self.decide(value) for value in probabilities)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    policy: SelectivePolicy
    validation: BinaryEvaluation


def calibrate_selective_policy(
    probabilities: Sequence[float],
    outcomes: Sequence[Prediction],
    *,
    min_entries: int = 100,
    min_coverage: float = 0.10,
) -> CalibrationResult:
    """Choose a threshold on validation only, favoring statistical support.

    The objective is the Wilson 95% lower bound, then raw accuracy, then coverage.
    This prevents a tiny handful of lucky bets from winning threshold selection.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if min_entries < 1:
        raise ValueError("min_entries must be positive")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")

    confidences = sorted({round(abs(float(p) - 0.5) * 2.0, 8) for p in probabilities})
    candidates = (0.0, *confidences)
    best: tuple[tuple[float, float, float, float], SelectivePolicy, BinaryEvaluation] | None = None

    for threshold in candidates:
        policy = SelectivePolicy(threshold)
        predictions = policy.apply(probabilities)
        metrics = evaluate_binary("validation", predictions, outcomes, p_up=probabilities)
        if metrics.entered < min_entries or metrics.coverage < min_coverage:
            continue
        lower, _upper = metrics.wilson_95
        key = (lower, metrics.accuracy, metrics.coverage, -threshold)
        if best is None or key > best[0]:
            best = (key, policy, metrics)

    if best is None:
        raise ValueError("no threshold satisfies min_entries/min_coverage")
    return CalibrationResult(policy=best[1], validation=best[2])
