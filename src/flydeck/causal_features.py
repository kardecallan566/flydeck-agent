from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CausalFeatureVector:
    values: tuple[float, ...]
    short_return: float
    medium_return: float
    long_return: float
    volatility: float
    volume_relative: float
    novelty: float


class CausalFeatureBank:
    """Small, causal, multiscale feature bank with online robust normalization."""

    def __init__(self, horizons: tuple[int, ...] = (2, 6, 12, 24, 48, 96), alpha: float = 0.02) -> None:
        if not horizons or any(h < 1 for h in horizons) or not 0.0 < alpha <= 1.0:
            raise ValueError("invalid feature-bank configuration")
        self.horizons = horizons
        self.alpha = alpha
        self._mean: list[float] = []
        self._variance: list[float] = []
        self._initialized = False

    def reset(self) -> None:
        self._mean.clear()
        self._variance.clear()
        self._initialized = False

    def transform(self, prices: tuple[float, ...], volumes: tuple[float, ...] | None = None) -> CausalFeatureVector:
        if len(prices) < 2 or any(value <= 0.0 for value in prices):
            raise ValueError("prices must contain at least two positive values")
        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        raw: list[float] = []
        horizon_returns: list[float] = []
        horizon_vols: list[float] = []
        for horizon in self.horizons:
            count = min(horizon, len(returns))
            window = returns[-count:]
            total = sum(window)
            mean = total / max(1, count)
            variance = sum((value - mean) ** 2 for value in window) / max(1, count)
            horizon_returns.append(total)
            horizon_vols.append(math.sqrt(max(0.0, variance)))
            raw.extend((total, mean, math.sqrt(max(0.0, variance))))
        latest = returns[-1]
        acceleration = returns[-1] - returns[-2] if len(returns) > 1 else 0.0
        persistence = sum(a * b > 0.0 for a, b in zip(returns, returns[1:])) / max(1, len(returns) - 1)
        volume_relative = 1.0
        if volumes and len(volumes) >= 2:
            vwindow = volumes[-min(len(volumes), max(self.horizons)):]
            mean_volume = sum(vwindow) / max(1, len(vwindow))
            volume_relative = volumes[-1] / max(1e-12, mean_volume)
        volume_signal = max(-1.0, min(1.0, math.log(max(1e-12, volume_relative))))
        raw.extend((latest, acceleration, persistence, volume_signal))

        normalized = self._normalize(raw)
        short = normalized[0] if normalized else 0.0
        medium_index = min(3 * 2, len(normalized) - 1)
        long_index = min(3 * 4, len(normalized) - 1)
        volatility = max(0.0, min(1.0, horizon_vols[0] * 100.0 if horizon_vols else 0.0))
        novelty = max(0.0, min(1.0, abs(normalized[-3]) / 3.0 if len(normalized) >= 3 else 0.0))
        return CausalFeatureVector(
            values=tuple(normalized),
            short_return=short,
            medium_return=normalized[medium_index],
            long_return=normalized[long_index],
            volatility=volatility,
            volume_relative=volume_signal,
            novelty=novelty,
        )

    def _normalize(self, values: list[float]) -> list[float]:
        if not self._initialized:
            self._mean = list(values)
            self._variance = [1e-6] * len(values)
            self._initialized = True
            return [0.0] * len(values)
        if len(values) != len(self._mean):
            self._mean = [0.0] * len(values)
            self._variance = [1e-6] * len(values)
        result: list[float] = []
        for index, value in enumerate(values):
            delta = value - self._mean[index]
            self._mean[index] += self.alpha * delta
            self._variance[index] = (1.0 - self.alpha) * self._variance[index] + self.alpha * delta * delta
            scale = math.sqrt(max(1e-8, self._variance[index]))
            result.append(max(-4.0, min(4.0, delta / scale)))
        return result
