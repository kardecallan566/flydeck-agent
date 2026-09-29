from __future__ import annotations

import math
from typing import Sequence

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset


def causal_feature_vector(dataset: BNBPredictionDataset, index: int) -> tuple[float, ...]:
    """Causal multi-horizon OHLCV features using candle index and older data only."""
    if not 0 <= index < dataset.size - 1:
        raise IndexError("feature index requires a following target candle")

    close = dataset.closes[index]
    open_price = dataset.opens[index]
    high = dataset.highs[index]
    low = dataset.lows[index]

    def ret(lag: int) -> float:
        j = max(0, index - lag)
        base = dataset.closes[j]
        return close / max(1e-12, base) - 1.0

    def return_stats(window: int) -> tuple[float, float]:
        start_i = max(1, index - window + 1)
        values = [
            dataset.closes[i] / max(1e-12, dataset.closes[i - 1]) - 1.0
            for i in range(start_i, index + 1)
        ]
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(max(0.0, variance))

    def mean_slice(values: Sequence[float], window: int) -> float:
        start_i = max(0, index - window + 1)
        chunk = values[start_i:index + 1]
        return sum(chunk) / len(chunk) if chunk else 0.0

    def ema_close(window: int) -> float:
        start_i = max(0, index - window * 3 + 1)
        alpha = 2.0 / (window + 1.0)
        ema = dataset.closes[start_i]
        for value in dataset.closes[start_i + 1:index + 1]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    mean12, vol12 = return_stats(12)
    _mean24, vol24 = return_stats(24)
    current_range = (high - low) / max(1e-12, close)
    body = (close - open_price) / max(1e-12, open_price)
    candle_span = max(1e-12, high - low)
    upper_wick = (high - max(open_price, close)) / candle_span
    lower_wick = (min(open_price, close) - low) / candle_span

    range_start = max(0, index - 11)
    ranges = [
        (dataset.highs[i] - dataset.lows[i]) / max(1e-12, dataset.closes[i])
        for i in range(range_start, index + 1)
    ]
    mean_range12 = sum(ranges) / len(ranges) if ranges else current_range
    range_surprise = current_range / max(1e-12, mean_range12) - 1.0

    mean_volume24 = mean_slice(dataset.volumes, 24)
    mean_volume6 = mean_slice(dataset.volumes, 6)
    volume_ratio24 = math.log(max(1e-12, dataset.volumes[index]) / max(1e-12, mean_volume24))
    volume_ratio6 = math.log(max(1e-12, dataset.volumes[index]) / max(1e-12, mean_volume6))
    volume_trend = mean_volume6 / max(1e-12, mean_volume24) - 1.0

    ema6 = ema_close(6)
    ema12 = ema_close(12)
    ema_gap6 = close / max(1e-12, ema6) - 1.0
    ema_gap12 = close / max(1e-12, ema12) - 1.0

    position_start = max(0, index - 23)
    rolling_high = max(dataset.highs[position_start:index + 1])
    rolling_low = min(dataset.lows[position_start:index + 1])
    range24 = max(1e-12, rolling_high - rolling_low)
    range_position = (close - rolling_low) / range24 - 0.5

    minute_of_day = (dataset.timestamps[index] // 60_000) % (24 * 60)
    phase = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)

    return (
        ret(1),
        ret(2),
        ret(3),
        ret(6),
        ret(12),
        ret(24),
        mean12,
        vol12,
        vol24,
        current_range,
        body,
        upper_wick,
        lower_wick,
        range_surprise,
        volume_ratio24,
        volume_ratio6,
        volume_trend,
        dataset.trend_persistence(index, window=12) - 0.5,
        dataset.trend_persistence(index, window=24) - 0.5,
        ema_gap6,
        ema_gap12,
        range_position,
        math.sin(phase),
        math.cos(phase),
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
        return tuple(max(-8.0, min(8.0, (value - mean) / scale)) for value, mean, scale in zip(row, self.means, self.scales))


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
