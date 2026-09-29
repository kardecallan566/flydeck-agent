from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bnb_prediction import Prediction
from .statistical_metrics import BinaryEvaluation, evaluate_binary


@dataclass(frozen=True, slots=True)
class SelectivePolicy:
    """Frozen asymmetric UP/DOWN/WAIT rule calibrated on validation only."""

    confidence_threshold: float = 0.0
    up_confidence_threshold: float | None = None
    down_confidence_threshold: float | None = None

    @property
    def up_threshold(self) -> float:
        return self.confidence_threshold if self.up_confidence_threshold is None else self.up_confidence_threshold

    @property
    def down_threshold(self) -> float:
        return self.confidence_threshold if self.down_confidence_threshold is None else self.down_confidence_threshold

    def decide(self, p_up: float) -> Prediction:
        probability = min(1.0, max(0.0, float(p_up)))
        confidence = abs(probability - 0.5) * 2.0
        if probability >= 0.5:
            return Prediction.UP if confidence >= self.up_threshold else Prediction.WAIT
        return Prediction.DOWN if confidence >= self.down_threshold else Prediction.WAIT

    def apply(self, probabilities: Sequence[float]) -> tuple[Prediction, ...]:
        return tuple(self.decide(value) for value in probabilities)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    policy: SelectivePolicy
    validation: BinaryEvaluation


def _candidate_thresholds(
    probabilities: Sequence[float],
    *,
    up: bool,
    max_candidates: int = 25,
) -> tuple[float, ...]:
    values = sorted({
        round(abs(float(p) - 0.5) * 2.0, 8)
        for p in probabilities
        if (float(p) >= 0.5) == up
    })
    if not values:
        return (0.0,)
    if len(values) <= max_candidates:
        return tuple(sorted({0.0, *values}))
    sampled = {0.0}
    for i in range(max_candidates):
        position = round(i * (len(values) - 1) / max(1, max_candidates - 1))
        sampled.add(values[position])
    return tuple(sorted(sampled))


def calibrate_selective_policy(
    probabilities: Sequence[float],
    outcomes: Sequence[Prediction],
    *,
    min_entries: int = 100,
    min_coverage: float = 0.10,
    asymmetric: bool = True,
    max_candidates_per_side: int = 25,
) -> CalibrationResult:
    """Select validation-only abstention thresholds with support constraints.

    The objective remains conservative: Wilson 95% lower bound first, then
    accuracy, balanced accuracy and coverage. With asymmetric selection the UP
    and DOWN tails receive independent thresholds, allowing the policy to
    abstain more aggressively on a poorly calibrated direction. Candidate
    count is bounded to reduce threshold overfitting.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if min_entries < 1:
        raise ValueError("min_entries must be positive")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    if max_candidates_per_side < 2:
        raise ValueError("max_candidates_per_side must be at least 2")

    if asymmetric:
        up_candidates = _candidate_thresholds(
            probabilities, up=True, max_candidates=max_candidates_per_side
        )
        down_candidates = _candidate_thresholds(
            probabilities, up=False, max_candidates=max_candidates_per_side
        )
    else:
        all_candidates = tuple(sorted(set(
            _candidate_thresholds(probabilities, up=True, max_candidates=max_candidates_per_side)
            + _candidate_thresholds(probabilities, up=False, max_candidates=max_candidates_per_side)
        )))
        up_candidates = down_candidates = all_candidates

    best: tuple[tuple[float, float, float, float, float], SelectivePolicy, BinaryEvaluation] | None = None
    for up_threshold in up_candidates:
        for down_threshold in down_candidates:
            if not asymmetric and up_threshold != down_threshold:
                continue
            policy = SelectivePolicy(
                confidence_threshold=0.0,
                up_confidence_threshold=up_threshold,
                down_confidence_threshold=down_threshold,
            )
            predictions = policy.apply(probabilities)
            metrics = evaluate_binary("validation", predictions, outcomes, p_up=probabilities)
            if metrics.entered < min_entries or metrics.coverage < min_coverage:
                continue
            lower, _upper = metrics.wilson_95
            key = (
                lower,
                metrics.accuracy,
                metrics.balanced_accuracy,
                metrics.coverage,
                -(up_threshold + down_threshold),
            )
            if best is None or key > best[0]:
                best = (key, policy, metrics)

    if best is None:
        raise ValueError("no threshold satisfies min_entries/min_coverage")
    return CalibrationResult(policy=best[1], validation=best[2])
