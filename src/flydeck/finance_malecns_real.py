from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from .finance import Candle, CryptoTradingEnvironment
from .malecns import MaleCNSCircuit, MaleCNSReservoir


@dataclass(frozen=True, slots=True)
class RealMarketDataset:
    symbol: str
    interval: str
    candles: tuple[Candle, ...]
    timestamps_ms: tuple[int, ...]
    source: str

    @property
    def rows(self) -> int:
        return len(self.candles)


@dataclass(frozen=True, slots=True)
class RealMarketSplits:
    train: RealMarketDataset
    validation: RealMarketDataset
    test: RealMarketDataset


@dataclass(frozen=True, slots=True)
class MaleCNSMarketResult:
    return_pct: float
    final_portfolio: float
    drawdown_pct: float
    trades: int
    steps: int
    action_counts: tuple[int, int, int]
    buy_hold_return_pct: float
    score_means: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MaleCNSRealBenchmark:
    train: MaleCNSMarketResult
    validation: MaleCNSMarketResult
    test: MaleCNSMarketResult
    random_test: MaleCNSMarketResult


def load_btcusdt_dataset(path: str | Path) -> RealMarketDataset:
    """Load a Binance-style OHLCV CSV without external data dependencies."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("market CSV has no header")
        names = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"open_time", "open", "high", "low", "close", "volume"}
        missing = required - names.keys()
        if missing:
            raise ValueError(f"market CSV is missing columns: {sorted(missing)}")
        rows = list(reader)

    candles: list[Candle] = []
    timestamps: list[int] = []
    for row in rows:
        timestamps.append(int(float(row[names["open_time"]])))
        candles.append(Candle(float(row[names["open"]]), float(row[names["high"]]),
                              float(row[names["low"]]), float(row[names["close"]]),
                              float(row[names["volume"]])))
    if len(candles) < 26:
        raise ValueError("market CSV must contain at least 26 rows")
    return RealMarketDataset("BTCUSDT", "1h", tuple(candles), tuple(timestamps), str(path))


def split_real_market(dataset: RealMarketDataset, *, train_ratio: float = 0.70,
                      validation_ratio: float = 0.15, context: int = 24) -> RealMarketSplits:
    """Split chronologically; validation/test retain context from the preceding split."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("invalid split ratios")
    n = dataset.rows
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    if train_end <= context or validation_end <= train_end:
        raise ValueError("dataset is too small for requested split/context")

    def make(start: int, end: int) -> RealMarketDataset:
        return RealMarketDataset(dataset.symbol, dataset.interval, dataset.candles[start:end],
                                 dataset.timestamps_ms[start:end], dataset.source)

    return RealMarketSplits(make(0, train_end), make(max(0, train_end - context), validation_end),
                            make(max(0, validation_end - context), n))


def run_malecns_real_benchmark(circuit: MaleCNSCircuit, dataset: RealMarketDataset, *, seed: int = 42,
                               train_ratio: float = 0.70, validation_ratio: float = 0.15,
                               context: int = 24, train_epsilon: float = 0.20,
                               train_epsilon_decay: float = 0.9999, min_epsilon: float = 0.05,
                               learning_rate: float = 0.005, discount: float = 0.97) -> MaleCNSRealBenchmark:
    splits = split_real_market(dataset, train_ratio=train_ratio, validation_ratio=validation_ratio, context=context)
    reservoir = MaleCNSReservoir(circuit, feature_count=12, action_count=3, seed=seed)
    rng = random.Random(seed)
    train = _train_split(reservoir, splits.train.candles, rng, epsilon=train_epsilon,
                         epsilon_decay=train_epsilon_decay, min_epsilon=min_epsilon,
                         learning_rate=learning_rate, discount=discount)
    validation = _evaluate_split(reservoir, splits.validation.candles)
    test = _evaluate_split(reservoir, splits.test.candles)
    random_test = _random_split(splits.test.candles, seed=seed)
    return MaleCNSRealBenchmark(train, validation, test, random_test)


def _train_split(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...], rng: random.Random, *,
                 epsilon: float, epsilon_decay: float, min_epsilon: float,
                 learning_rate: float, discount: float) -> MaleCNSMarketResult:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    counts = [0, 0, 0]
    total_reward = 0.0
    current_epsilon = epsilon
    for _ in range(environment.max_steps):
        action = _choose_action(scores, current_epsilon, rng)
        counts[action] += 1
        activity = reservoir.action_activity(action)
        result = environment.step(action)
        total_reward += result.reward
        next_scores = (0.0, 0.0, 0.0) if result.done else reservoir.step(result.observation)
        target = result.reward if result.done else result.reward + discount * max(next_scores)
        td_error = max(-1.0, min(1.0, target - scores[action]))
        reservoir.update_readout(action, td_error, learning_rate, activity)
        scores = next_scores
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        if result.done:
            break
    return _result(environment, total_reward, candles, counts)


def _evaluate_split(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...]) -> MaleCNSMarketResult:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    counts = [0, 0, 0]
    totals = [0.0, 0.0, 0.0]
    total_reward = 0.0
    for _ in range(environment.max_steps):
        action = max(range(3), key=scores.__getitem__)
        counts[action] += 1
        for index in range(3):
            totals[index] += scores[index]
        result = environment.step(action)
        total_reward += result.reward
        if result.done:
            break
        scores = reservoir.step(result.observation)
    steps = max(1, environment.steps)
    return _result(environment, total_reward, candles, counts, tuple(v / steps for v in totals))


def _random_split(candles: tuple[Candle, ...], *, seed: int) -> MaleCNSMarketResult:
    environment = CryptoTradingEnvironment(candles)
    rng = random.Random(seed)
    counts = [0, 0, 0]
    total_reward = 0.0
    for _ in range(environment.max_steps):
        action = rng.randrange(3)
        counts[action] += 1
        result = environment.step(action)
        total_reward += result.reward
        if result.done:
            break
    return _result(environment, total_reward, candles, counts)


def _result(environment: CryptoTradingEnvironment, total_reward: float, candles: tuple[Candle, ...],
            counts: list[int], score_means: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> MaleCNSMarketResult:
    metrics = environment.episode_result(total_reward)
    buy_hold = candles[-1].close / candles[environment.window].close - 1.0
    return MaleCNSMarketResult(metrics.return_pct, metrics.final_portfolio, metrics.max_drawdown_pct,
                               metrics.trades, metrics.steps, tuple(counts), buy_hold * 100.0, score_means)


def _choose_action(scores: tuple[float, ...], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(scores))
    return max(range(len(scores)), key=scores.__getitem__)
