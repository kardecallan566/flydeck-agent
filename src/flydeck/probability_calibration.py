from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .bnb_prediction import Prediction


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """One-dimensional logistic calibrator fitted on held-out probabilities."""

    slope: float = 1.0
    intercept: float = 0.0

    def transform(self, probability: float) -> float:
        p = min(1.0 - 1e-6, max(1e-6, float(probability)))
        logit = math.log(p / (1.0 - p))
        return _sigmoid(self.intercept + self.slope * logit)

    def apply(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        return tuple(self.transform(value) for value in probabilities)


def fit_platt_calibrator(
    probabilities: Sequence[float],
    outcomes: Sequence[Prediction],
    *,
    learning_rate: float = 0.05,
    epochs: int = 300,
    l2: float = 1e-3,
) -> PlattCalibrator:
    """Fit Platt scaling on a chronological calibration slice only.

    WAIT/tie outcomes are ignored. The caller is responsible for ensuring this
    slice is strictly before any validation slice used to select an abstention
    policy.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")

    rows: list[tuple[float, float]] = []
    for probability, outcome in zip(probabilities, outcomes):
        if outcome not in (Prediction.UP, Prediction.DOWN):
            continue
        p = min(1.0 - 1e-6, max(1e-6, float(probability)))
        x = math.log(p / (1.0 - p))
        y = 1.0 if outcome == Prediction.UP else 0.0
        rows.append((x, y))
    if len(rows) < 20:
        return PlattCalibrator()

    slope = 1.0
    intercept = 0.0
    n = float(len(rows))
    for _ in range(max(1, epochs)):
        grad_slope = 0.0
        grad_intercept = 0.0
        for x, y in rows:
            pred = _sigmoid(intercept + slope * x)
            error = pred - y
            grad_slope += error * x
            grad_intercept += error
        slope -= learning_rate * (grad_slope / n + l2 * slope)
        intercept -= learning_rate * grad_intercept / n
        slope = max(-8.0, min(8.0, slope))
        intercept = max(-8.0, min(8.0, intercept))

    return PlattCalibrator(slope=slope, intercept=intercept)


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))
