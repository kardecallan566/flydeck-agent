from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bnb_prediction import Prediction
from .bnb_prediction_agent import BNBObservation, FlyBNBPredictionAgent, FlyDecision
from .data import BinanceMarketDataProvider, MarketCandle, MarketDataService, inspect_dataset
from .data.market_cache import write_csv
from .malecns import MaleCNSCircuit


@dataclass(frozen=True, slots=True)
class BNBPredictionDataset:
    """BNB-specific view over the canonical market-data contract."""

    timestamps: tuple[int, ...]
    opens: tuple[float, ...]
    highs: tuple[float, ...]
    lows: tuple[float, ...]
    closes: tuple[float, ...]
    volumes: tuple[float, ...]

    @classmethod
    def from_candles(cls, candles: tuple[MarketCandle, ...]) -> "BNBPredictionDataset":
        if not candles:
            raise ValueError("BNB dataset is empty")
        return cls(
            timestamps=tuple(c.timestamp for c in candles),
            opens=tuple(c.open for c in candles),
            highs=tuple(c.high for c in candles),
            lows=tuple(c.low for c in candles),
            closes=tuple(c.close for c in candles),
            volumes=tuple(c.volume for c in candles),
        )

    @property
    def size(self) -> int:
        return len(self.closes)

    def observation(self, index: int, context: int = 24) -> BNBObservation:
        start = max(0, index - context + 1)
        return BNBObservation(
            timestamp=self.timestamps[index],
            prices=self.closes[start : index + 1],
            volumes=self.volumes[start : index + 1],
        )

    def outcome(self, index: int) -> Prediction:
        if not 0 <= index < self.size - 1:
            raise IndexError("outcome requires an index with a following candle")
        if self.closes[index + 1] > self.closes[index]:
            return Prediction.UP
        if self.closes[index + 1] < self.closes[index]:
            return Prediction.DOWN
        return Prediction.WAIT

    def volatility(self, index: int, window: int = 24) -> float:
        """Normalized average true range over the preceding window."""
        start = max(0, index - window + 1)
        ranges = [
            (high - low) / max(1e-9, close)
            for high, low, close in zip(
                self.highs[start : index + 1],
                self.lows[start : index + 1],
                self.closes[start : index + 1],
            )
        ]
        return sum(ranges) / len(ranges) if ranges else 0.0

    def volatility_regime(self, index: int, window: int = 24, threshold: float = 0.0035) -> str:
        return "high" if self.volatility(index, window=window) >= threshold else "low"

    def trend_persistence(self, index: int, window: int = 24) -> float:
        start = max(0, index - window)
        window_closes = self.closes[start : index + 1]
        if len(window_closes) < 3:
            return 0.5
        diffs = [b - a for a, b in zip(window_closes, window_closes[1:])]
        same_dir = sum(a * b > 0 for a, b in zip(diffs, diffs[1:]))
        return same_dir / max(1, len(diffs) - 1)

    def trend_regime(self, index: int, window: int = 24, threshold: float = 0.45) -> str:
        return "trending" if self.trend_persistence(index, window=window) >= threshold else "chop"


def load_bnb_5m_csv(path: str | Path) -> BNBPredictionDataset:
    """Load legacy or canonical CSV through the shared market-data layer."""
    service = MarketDataService(BinanceMarketDataProvider())
    dataset = service.load(path, symbol="BNBUSDT", interval="5m")
    health = inspect_dataset(dataset)
    if not health.complete:
        raise ValueError(
            f"BNB data contains {health.gaps} interval gaps; repair the dataset before benchmarking"
        )
    return BNBPredictionDataset.from_candles(dataset.candles)


def download_bnb_5m_csv(path: str | Path, limit: int = 1000) -> Path:
    """Download BNB 5m candles through the provider/cache architecture."""
    destination = Path(path)
    service = MarketDataService(BinanceMarketDataProvider(), cache_root=destination.parent)
    dataset = service.fetch("BNBUSDT", "5m", limit=limit)
    write_csv(dataset, destination)
    return destination


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    rounds: int
    entered: int
    correct: int
    accuracy: float
    coverage: float
    up: int
    down: int
    wait: int


def _run_split(
    agent: FlyBNBPredictionAgent,
    data: BNBPredictionDataset,
    start: int,
    end: int,
    learn: bool,
) -> PredictionMetrics:
    entered = correct = up = down = wait = 0
    for index in range(start, end):
        prediction = agent.predict(data.observation(index))
        outcome = data.outcome(index)
        if prediction.decision == FlyDecision.UP:
            up += 1
            entered += 1
            correct += int(outcome == Prediction.UP)
        elif prediction.decision == FlyDecision.DOWN:
            down += 1
            entered += 1
            correct += int(outcome == Prediction.DOWN)
        else:
            wait += 1
        if learn:
            agent.learn(outcome)
    rounds = end - start
    return PredictionMetrics(
        rounds=rounds,
        entered=entered,
        correct=correct,
        accuracy=correct / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        up=up,
        down=down,
        wait=wait,
    )


def run_bnb_prediction_benchmark(
    data: BNBPredictionDataset,
    circuit: MaleCNSCircuit,
    seed: int = 123,
    confidence_threshold: float = 0.20,
) -> tuple[PredictionMetrics, PredictionMetrics, PredictionMetrics]:
    usable = data.size - 1
    if usable < 30:
        raise ValueError("dataset needs at least 31 candles")
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    agent = FlyBNBPredictionAgent(circuit, seed=seed, confidence_threshold=confidence_threshold)
    train = _run_split(agent, data, 24, train_end, True)
    validation = _run_split(agent, data, train_end, validation_end, False)
    test = _run_split(agent, data, validation_end, usable, False)
    return train, validation, test
