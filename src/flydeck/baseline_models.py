from __future__ import annotations

import math
from typing import Sequence

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset


def causal_feature_vector(dataset: BNBPredictionDataset, index: int) -> tuple[float, ...]:
    """Small OHLCV feature set using candle index and older data only."""
    if not 0 <= index < dataset.size - 1:
        raise IndexError("feature index requires a following target candle")

    close = dataset.closes[index]

    def ret(lag: int) -> float:
        j = max(0, index - lag)
        base = dataset.closes[j]
        return close / max(1e-12, base) - 1.0

    start = max(1, index - 11)
    returns = [
        dataset.closes[i] / max(1e-12, dataset.closes[i - 1]) - 1.0
        for i in range(start, index + 1)
    ]
    mean_ret = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean_ret) ** 2 for value in returns) / len(returns) if returns else 0.0
    current_range = (dataset.highs[index] - dataset.lows[index]) / max(1e-12, close)
    body = (dataset.closes[index] - dataset.opens[index]) / max(1e-12, dataset.opens[index])
    v_start = max(0, index - 23)
    volume_window = dataset.volumes[v_start:index + 1]
    mean_volume = sum(volume_window) / len(volume_window) if volume_window else 1.0
    volume_ratio = dataset.volumes[index] / max(1e-12, mean_volume) - 1.0

    return (
        ret(1),
        ret(3),
        ret(6),
        ret(12),
        mean_ret,
        math.sqrt(max(0.0, variance)),
        current_range,
        body,
        volume_ratio,
        dataset.trend_persistence(index, window=12) - 0.5,
    )


class _Standardizer:
    def __init__(self) -> None:
        self.means: tuple[float, ...] = ()
        self.scales: tuple[float, ...] = ()

    def fit(self, rows: Sequence[tuple[float, ...]]) -> None:
        if not rows:
            raise ValueError("cannot fit standardizer on empty rows")
        width = len(rows[0])
        means = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
        scales = []
        for j, mean in enumerate(means):
            variance = sum((row[j] - mean) ** 2 for row in rows) / len(rows)
            scales.append(max(1e-9, math.sqrt(variance)))
        self.means = tuple(means)
        self.scales = tuple(scales)

    def transform(self, row: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales))


class LogisticRegressionBaseline:
    """Dependency-free logistic regression trained only on the supplied train range."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 250, l2: float = 1e-3) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self._scaler = _Standardizer()
        self._weights: list[float] = []
        self._bias = 0.0

    def fit(self, dataset: BNBPredictionDataset, start: int, end: int) -> None:
        rows: list[tuple[float, ...]] = []
        labels: list[float] = []
        for index in range(start, end):
            outcome = dataset.outcome(index)
            if outcome not in (Prediction.UP, Prediction.DOWN):
                continue
            rows.append(causal_feature_vector(dataset, index))
            labels.append(1.0 if outcome == Prediction.UP else 0.0)
        if len(rows) < 20:
            raise ValueError("logistic baseline needs at least 20 binary training rows")
        self._scaler.fit(rows)
        x = [self._scaler.transform(row) for row in rows]
        self._weights = [0.0] * len(x[0])
        self._bias = 0.0
        n = len(x)
        for _ in range(self.epochs):
            grad_w = [0.0] * len(self._weights)
            grad_b = 0.0
            for row, label in zip(x, labels):
                p = _sigmoid(self._bias + sum(w * value for w, value in zip(self._weights, row)))
                error = p - label
                grad_b += error
                for j, value in enumerate(row):
                    grad_w[j] += error * value
            for j in range(len(self._weights)):
                gradient = grad_w[j] / n + self.l2 * self._weights[j]
                self._weights[j] -= self.learning_rate * gradient
            self._bias -= self.learning_rate * grad_b / n

    def predict_probability(self, dataset: BNBPredictionDataset, index: int) -> float:
        if not self._weights:
            raise RuntimeError("fit must be called before predict")
        row = self._scaler.transform(causal_feature_vector(dataset, index))
        return _sigmoid(self._bias + sum(w * value for w, value in zip(self._weights, row)))

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        return Prediction.UP if self.predict_probability(dataset, index) >= 0.5 else Prediction.DOWN


class AdaBoostStumpBaseline:
    """Tiny tabular ML baseline: AdaBoost over one-dimensional decision stumps."""

    def __init__(self, estimators: int = 12) -> None:
        self.estimators = estimators
        self._scaler = _Standardizer()
        self._stumps: list[tuple[int, float, int, float]] = []

    def fit(self, dataset: BNBPredictionDataset, start: int, end: int) -> None:
        rows: list[tuple[float, ...]] = []
        labels: list[int] = []
        for index in range(start, end):
            outcome = dataset.outcome(index)
            if outcome not in (Prediction.UP, Prediction.DOWN):
                continue
            rows.append(causal_feature_vector(dataset, index))
            labels.append(1 if outcome == Prediction.UP else -1)
        if len(rows) < 20:
            raise ValueError("AdaBoost baseline needs at least 20 binary training rows")
        self._scaler.fit(rows)
        x = [self._scaler.transform(row) for row in rows]
        weights = [1.0 / len(x)] * len(x)
        self._stumps = []

        for _ in range(self.estimators):
            best: tuple[float, int, float, int] | None = None
            for feature in range(len(x[0])):
                values = sorted(row[feature] for row in x)
                thresholds = _quantile_thresholds(values)
                for threshold in thresholds:
                    for polarity in (1, -1):
                        error = 0.0
                        for sample_weight, row, label in zip(weights, x, labels):
                            pred = polarity if row[feature] >= threshold else -polarity
                            if pred != label:
                                error += sample_weight
                        if best is None or error < best[0]:
                            best = (error, feature, threshold, polarity)
            if best is None:
                break
            error, feature, threshold, polarity = best
            error = min(1.0 - 1e-9, max(1e-9, error))
            if error >= 0.5:
                break
            alpha = 0.5 * math.log((1.0 - error) / error)
            self._stumps.append((feature, threshold, polarity, alpha))
            normalizer = 0.0
            for i, (row, label) in enumerate(zip(x, labels)):
                pred = polarity if row[feature] >= threshold else -polarity
                weights[i] *= math.exp(-alpha * label * pred)
                normalizer += weights[i]
            weights = [value / max(1e-12, normalizer) for value in weights]

    def predict_probability(self, dataset: BNBPredictionDataset, index: int) -> float:
        if not self._stumps:
            return 0.5
        row = self._scaler.transform(causal_feature_vector(dataset, index))
        score = 0.0
        for feature, threshold, polarity, alpha in self._stumps:
            pred = polarity if row[feature] >= threshold else -polarity
            score += alpha * pred
        return _sigmoid(2.0 * score)

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        return Prediction.UP if self.predict_probability(dataset, index) >= 0.5 else Prediction.DOWN


def _quantile_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return (0.0,)
    positions = (0.10, 0.25, 0.50, 0.75, 0.90)
    return tuple(values[min(len(values) - 1, int((len(values) - 1) * q))] for q in positions)


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))
