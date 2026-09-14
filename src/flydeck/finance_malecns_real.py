from __future__ import annotations

import random
from dataclasses import dataclass

from .finance import Candle, CryptoTradingEnvironment
from .finance_data import RealMarketDataset, load_ohlcv_csv
from .finance_real import split_real_market
from .malecns import MaleCNSCircuit, MaleCNSReservoir


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


def load_btcusdt_dataset(path: str) -> RealMarketDataset:
    return load_ohlcv_csv(path, symbol="BTCUSDT", interval="1h")


def run_malecns_real_benchmark(
    circuit: MaleCNSCircuit,
    dataset: RealMarketDataset,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    context: int = 24,
    train_epsilon: float = 0.20,
    train_epsilon_decay: float = 0.9999,
    min_epsilon: float = 0.05,
    learning_rate: float = 0.005,
    discount: float = 0.97,
) -> MaleCNSRealBenchmark:
    splits = split_real_market(
        dataset,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        context=context,
    )
    reservoir = MaleCNSReservoir(circuit, feature_count=12, action_count=3, seed=seed)
    rng = random.Random(seed)

    train = _train_split(
        reservoir,
        splits.train.candles,
        rng,
        epsilon=train_epsilon,
        epsilon_decay=train_epsilon_decay,
        min_epsilon=min_epsilon,
        learning_rate=learning_rate,
        discount=discount,
    )
    validation = _evaluate_split(reservoir, splits.validation.candles)
    test = _evaluate_split(reservoir, splits.test.candles)
    random_test = _random_split(splits.test.candles, seed=seed)
    return MaleCNSRealBenchmark(train, validation, test, random_test)


def _train_split(
    reservoir: MaleCNSReservoir,
    candles: tuple[Candle, ...],
    rng: random.Random,
    *,
    epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    learning_rate: float,
    discount: float,
) -> MaleCNSMarketResult:
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
        if result.done:
            next_scores = (0.0, 0.0, 0.0)
        else:
            next_scores = reservoir.step(result.observation)
        target = result.reward if result.done else result.reward + discount * max(next_scores)
        td_error = max(-1.0, min(1.0, target - scores[action]))
        reservoir.update_readout(action, td_error, learning_rate, activity)
        scores = next_scores
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        if result.done:
            break

    return _result(environment, total_reward, candles, counts)


def _evaluate_split(
    reservoir: MaleCNSReservoir,
    candles: tuple[Candle, ...],
) -> MaleCNSMarketResult:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    counts = [0, 0, 0]
    score_totals = [0.0, 0.0, 0.0]
    total_reward = 0.0

    for _ in range(environment.max_steps):
        action = max(range(3), key=scores.__getitem__)
        counts[action] += 1
        for index in range(3):
            score_totals[index] += scores[index]
        result = environment.step(action)
        total_reward += result.reward
        if result.done:
            break
        scores = reservoir.step(result.observation)

    return _result(environment, total_reward, candles, counts, tuple(v / max(1, environment.steps) for v in score_totals))


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


def _result(
    environment: CryptoTradingEnvironment,
    total_reward: float,
    candles: tuple[Candle, ...],
    counts: list[int],
    score_means: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MaleCNSMarketResult:
    metrics = environment.episode_result(total_reward)
    buy_hold = candles[-1].close / candles[environment.window].close - 1.0
    return MaleCNSMarketResult(
        return_pct=metrics.return_pct,
        final_portfolio=metrics.final_portfolio,
        drawdown_pct=metrics.max_drawdown_pct,
        trades=metrics.trades,
        steps=metrics.steps,
        action_counts=tuple(counts),
        buy_hold_return_pct=buy_hold * 100.0,
        score_means=score_means,
    )


def _choose_action(scores: tuple[float, ...], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(scores))
    return max(range(len(scores)), key=scores.__getitem__)
