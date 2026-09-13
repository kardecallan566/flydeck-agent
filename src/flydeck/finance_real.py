from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .agent import Agent
from .finance import Candle, CryptoTradingEnvironment
from .finance_data import RealMarketDataset
from .finance_encoder import SparseMarketEncoder


@dataclass(frozen=True, slots=True)
class MarketSplits:
    train: RealMarketDataset
    validation: RealMarketDataset
    test: RealMarketDataset
    context: int


@dataclass(frozen=True, slots=True)
class RealBenchmarkResult:
    return_pct: float
    final_portfolio: float
    drawdown_pct: float
    trades: int
    steps: int
    action_counts: tuple[int, int, int]
    score_means: tuple[float, float, float]
    buy_hold_return_pct: float
    max_buy_streak: int
    average_holding_steps: float

    @property
    def trade_frequency(self) -> float:
        return self.trades / max(1, self.steps)

    @property
    def hold_rate(self) -> float:
        return self.action_counts[0] / max(1, self.steps)

    @property
    def buy_rate(self) -> float:
        return self.action_counts[1] / max(1, self.steps)

    @property
    def sell_rate(self) -> float:
        return self.action_counts[2] / max(1, self.steps)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    index: int
    action: int
    price: float
    position_ratio: float
    scores: tuple[float, float, float]
    future_returns: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class DecisionQuality:
    records: tuple[DecisionRecord, ...]
    action_counts: tuple[int, int, int]
    average_future_returns_by_action: tuple[tuple[float, ...], ...]
    average_buy_advantage: float
    average_sell_advantage: float
    max_buy_streak: int
    average_holding_steps: float


def split_real_market(
    dataset: RealMarketDataset,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    context: int = 24,
) -> MarketSplits:
    """Chronologically split data while retaining lookback context at boundaries."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("train_ratio and validation_ratio must be within [0, 1]")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1")
    if context < 1:
        raise ValueError("context must be >= 1")

    total = dataset.rows
    train_end = int(total * train_ratio)
    validation_end = int(total * (train_ratio + validation_ratio))
    if train_end <= context or validation_end <= train_end or total - validation_end <= 1:
        raise ValueError("dataset is too small for the requested chronological split")

    def make_slice(start: int, end: int, name: str) -> RealMarketDataset:
        actual_start = max(0, start - context)
        return RealMarketDataset(
            symbol=dataset.symbol,
            interval=dataset.interval,
            candles=dataset.candles[actual_start:end],
            timestamps_ms=dataset.timestamps_ms[actual_start:end],
            source=f"{dataset.source}:{name}",
        )

    return MarketSplits(
        train=make_slice(0, train_end, "train"),
        validation=make_slice(train_end, validation_end, "validation"),
        test=make_slice(validation_end, total, "test"),
        context=context,
    )


def future_returns(candles: tuple[Candle, ...], index: int, horizons: tuple[int, ...] = (1, 3, 6, 12, 24)) -> tuple[tuple[int, float], ...]:
    price = candles[index].close
    result: list[tuple[int, float]] = []
    for horizon in horizons:
        target = index + horizon
        if target < len(candles):
            result.append((horizon, candles[target].close / price - 1.0))
    return tuple(result)


def collect_decision_quality(
    agent: Agent,
    candles: tuple[Candle, ...],
    *,
    max_steps: int | None = None,
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
) -> DecisionQuality:
    """Run greedy decisions and attach forward market returns to every decision."""
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("real finance benchmark requires a 27-input / 3-action agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    records: list[DecisionRecord] = []
    counts = [0, 0, 0]
    buy_streak = 0
    max_buy_streak = 0
    holding_durations: list[int] = []
    current_holding = 0

    for _ in range(environment.max_steps):
        index = environment.index
        action = agent.choose_action(scores)
        counts[action] += 1
        records.append(
            DecisionRecord(
                index=index,
                action=action,
                price=candles[index].close,
                position_ratio=environment.position_ratio,
                scores=tuple(scores),
                future_returns=future_returns(candles, index, horizons),
            )
        )

        if action == 1:
            buy_streak += 1
            max_buy_streak = max(max_buy_streak, buy_streak)
        else:
            buy_streak = 0

        if environment.position_ratio > 0:
            current_holding += 1
        elif current_holding:
            holding_durations.append(current_holding)
            current_holding = 0

        result = environment.step(action)
        encoder.observe_action(action)
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    if current_holding:
        holding_durations.append(current_holding)

    average_by_action: list[tuple[float, ...]] = []
    for action in range(3):
        action_records = [record for record in records if record.action == action]
        horizon_values: list[float] = []
        for horizon in horizons:
            values = [dict(record.future_returns).get(horizon) for record in action_records]
            values = [value for value in values if value is not None]
            horizon_values.append(mean(values) if values else 0.0)
        average_by_action.append(tuple(horizon_values))

    buy_advantages = [record.scores[1] - record.scores[0] for record in records]
    sell_advantages = [record.scores[2] - record.scores[0] for record in records]
    return DecisionQuality(
        records=tuple(records),
        action_counts=tuple(counts),
        average_future_returns_by_action=tuple(average_by_action),
        average_buy_advantage=mean(buy_advantages) if buy_advantages else 0.0,
        average_sell_advantage=mean(sell_advantages) if sell_advantages else 0.0,
        max_buy_streak=max_buy_streak,
        average_holding_steps=mean(holding_durations) if holding_durations else 0.0,
    )


def train_real_market_v8(
    agent: Agent,
    candles: tuple[Candle, ...],
    *,
    max_steps: int | None = None,
    epsilon: float = 0.30,
    epsilon_decay: float = 0.9999,
    min_epsilon: float = 0.05,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
    discount: float = 0.97,
    trace_decay: float = 0.85,
) -> RealBenchmarkResult:
    """Train V8 through one chronological real-market pass.

    Unlike the synthetic episode trainer, the network is reset only once so a long
    historical dataset becomes one continuous learning stream instead of a sequence
    of short resets. No future/test data is touched by this function.
    """
    environment = CryptoTradingEnvironment(
        candles,
        max_steps=max_steps,
        trade_penalty=trade_penalty,
        invalid_action_penalty=invalid_action_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("real finance benchmark requires a 27-input / 3-action agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    counts = [0, 0, 0]
    total_reward = 0.0
    current_epsilon = epsilon

    for _ in range(environment.max_steps):
        action = agent.choose_action(scores)
        if agent._rng.random() < current_epsilon:
            action = agent._rng.randrange(3)
        counts[action] += 1
        result = environment.step(action)
        encoder.observe_action(action)
        if result.done:
            next_scores = (0.0, 0.0, 0.0)
        else:
            next_scores = agent.observe(encoder.encode(result.observation))
        agent.network.learn_td_selected_action(
            action,
            result.reward,
            next_scores,
            result.done,
            agent.learning_rate,
            discount,
            trace_decay,
        )
        total_reward += result.reward
        scores = next_scores
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        if result.done:
            break

    metrics = environment.episode_result(total_reward)
    buy_hold = candles[-1].close / candles[environment.window].close - 1.0
    return RealBenchmarkResult(
        return_pct=metrics.return_pct,
        final_portfolio=metrics.final_portfolio,
        drawdown_pct=metrics.max_drawdown_pct,
        trades=metrics.trades,
        steps=metrics.steps,
        action_counts=tuple(counts),
        score_means=(0.0, 0.0, 0.0),
        buy_hold_return_pct=buy_hold * 100.0,
        max_buy_streak=0,
        average_holding_steps=0.0,
    )


def evaluate_real_market(agent: Agent, candles: tuple[Candle, ...], *, max_steps: int | None = None) -> RealBenchmarkResult:
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("real finance benchmark requires a 27-input / 3-action agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    counts = [0, 0, 0]
    score_totals = [0.0, 0.0, 0.0]
    total_reward = 0.0

    for _ in range(environment.max_steps):
        action = agent.choose_action(scores)
        counts[action] += 1
        for index in range(3):
            score_totals[index] += scores[index]
        result = environment.step(action)
        encoder.observe_action(action)
        total_reward += result.reward
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    metrics = environment.episode_result(total_reward)
    buy_hold = candles[-1].close / candles[environment.window].close - 1.0
    denominator = max(1, metrics.steps)
    quality = collect_decision_quality(agent, candles, max_steps=max_steps)
    return RealBenchmarkResult(
        return_pct=metrics.return_pct,
        final_portfolio=metrics.final_portfolio,
        drawdown_pct=metrics.max_drawdown_pct,
        trades=metrics.trades,
        steps=metrics.steps,
        action_counts=tuple(counts),
        score_means=tuple(value / denominator for value in score_totals),
        buy_hold_return_pct=buy_hold * 100.0,
        max_buy_streak=quality.max_buy_streak,
        average_holding_steps=quality.average_holding_steps,
    )
