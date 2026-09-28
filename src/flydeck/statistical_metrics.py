from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .bnb_prediction import Prediction


@dataclass(frozen=True, slots=True)
class BinaryEvaluation:
    name: str
    total_rounds: int
    eligible_rounds: int
    entered: int
    correct: int
    true_up: int
    false_up: int
    true_down: int
    false_down: int
    brier_score: float | None

    @property
    def accuracy(self) -> float:
        return self.correct / self.entered if self.entered else 0.0

    @property
    def coverage(self) -> float:
        return self.entered / self.eligible_rounds if self.eligible_rounds else 0.0

    @property
    def recall_up(self) -> float:
        total = self.true_up + self.false_down
        return self.true_up / total if total else 0.0

    @property
    def recall_down(self) -> float:
        total = self.true_down + self.false_up
        return self.true_down / total if total else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall_up + self.recall_down)

    @property
    def wilson_95(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.entered)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate_binary(
    name: str,
    predictions: Sequence[Prediction],
    outcomes: Sequence[Prediction],
    *,
    p_up: Sequence[float] | None = None,
) -> BinaryEvaluation:
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have equal length")
    if p_up is not None and len(p_up) != len(outcomes):
        raise ValueError("p_up and outcomes must have equal length")

    entered = correct = true_up = false_up = true_down = false_down = 0
    eligible = 0
    brier_total = 0.0
    brier_count = 0
    for i, (pred, outcome) in enumerate(zip(predictions, outcomes)):
        if outcome not in (Prediction.UP, Prediction.DOWN):
            continue
        eligible += 1
        if p_up is not None:
            probability = min(1.0, max(0.0, float(p_up[i])))
            target = 1.0 if outcome == Prediction.UP else 0.0
            brier_total += (probability - target) ** 2
            brier_count += 1
        if pred == Prediction.WAIT:
            continue
        entered += 1
        if pred == Prediction.UP:
            if outcome == Prediction.UP:
                true_up += 1
                correct += 1
            else:
                false_up += 1
        elif pred == Prediction.DOWN:
            if outcome == Prediction.DOWN:
                true_down += 1
                correct += 1
            else:
                false_down += 1

    return BinaryEvaluation(
        name=name,
        total_rounds=len(outcomes),
        eligible_rounds=eligible,
        entered=entered,
        correct=correct,
        true_up=true_up,
        false_up=false_up,
        true_down=true_down,
        false_down=false_down,
        brier_score=brier_total / brier_count if brier_count else None,
    )
