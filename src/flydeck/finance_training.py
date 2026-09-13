from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .agent import Agent
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket


@dataclass(frozen=True, slots=True)
class FinanceTrainingResult:
    episodes: int
    average_return_pct: float
    best_return_pct: float
    last_return_pct: float
    average_drawdown_pct: float
    total_trades: int
    hold_actions: int
    buy_actions: int
    sell_actions: int

    @property
    def total_actions(self) -> int:
        return self.hold_actions + self.buy_actions + self.sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.total_trades / max(1, self.total_actions)


@dataclass(frozen=True, slots=True)
class FinanceEvaluationResult:
    return_pct: float
    final_portfolio: float
    max_drawdown_pct: float
    trades: int
    hold_actions: int
    buy_actions: int
    sell_actions: int

    @property
    def total_actions(self) -> int:
        return self.hold_actions + self.buy_actions + self.sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.trades / max(1, self.total_actions)


@dataclass(frozen=True, slots=True)
class FinanceBaselineResult:
    name: str
    return_pct: float
    final_portfolio: float
    max_drawdown_pct: float
    trades: int


@dataclass(frozen=True, slots=True)
class FinanceMultiMarketResult:
    markets: int
    average_return_pct: float
    median_return_pct: float
    average_drawdown_pct: float
    average_trades: float
    average_trade_frequency: float
    win_rate_vs_buy_hold: float
    average_excess_return_pct: float
    hold_average_return_pct: float
    buy_hold_average_return_pct: float


def _run_policy(candles: tuple, policy, max_steps: int) -> FinanceBaselineResult:
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    observation = environment.reset()
    total_reward = 0.0
    for step in range(max_steps):
        result = environment.step(policy(step, observation))
        total_reward += result.reward
        observation = result.observation
        if result.done:
            break
    metrics = environment.episode_result(total_reward)
    return FinanceBaselineResult("policy", metrics.return_pct, metrics.final_portfolio, metrics.max_drawdown_pct, metrics.trades)


def evaluate_hold(seed: int = 10_000, market_length: int = 256, max_steps: int = 200) -> FinanceBaselineResult:
    """Evaluate the no-risk HOLD baseline on a fresh market."""
    candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
    result = _run_policy(candles, lambda _step, _observation: 0, max_steps)
    return FinanceBaselineResult("HOLD", result.return_pct, result.final_portfolio, result.max_drawdown_pct, result.trades)


def evaluate_buy_and_hold(seed: int = 10_000, market_length: int = 256, max_steps: int = 200) -> FinanceBaselineResult:
    """Buy once, then hold the position for the rest of the market."""
    candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
    result = _run_policy(candles, lambda step, _observation: 1 if step == 0 else 0, max_steps)
    return FinanceBaselineResult("BUY & HOLD", result.return_pct, result.final_portfolio, result.max_drawdown_pct, result.trades)


def train_synthetic_crypto(
    agent: Agent,
    episodes: int = 100,
    market_length: int = 256,
    max_steps: int = 200,
    seed: int = 42,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.03,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
    discount: float = 0.97,
    trace_decay: float = 0.85,
) -> FinanceTrainingResult:
    """Train across generated markets using sparse TD learning with eligibility traces."""
    if episodes < 1 or market_length < 32 or max_steps < 1:
        raise ValueError("episodes must be >= 1, market_length must be >= 32 and max_steps must be >= 1")
    if not 0.0 <= epsilon <= 1.0 or not 0.0 < epsilon_decay <= 1.0 or not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("invalid epsilon configuration")
    if not 0.0 < discount <= 1.0 or not 0.0 <= trace_decay <= 1.0:
        raise ValueError("invalid TD configuration")

    returns, drawdowns = [], []
    total_trades, action_counts = 0, [0, 0, 0]
    current_epsilon = epsilon

    for episode in range(episodes):
        candles = SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate()
        environment = CryptoTradingEnvironment(
            candles,
            max_steps=max_steps,
            trade_penalty=trade_penalty,
            invalid_action_penalty=invalid_action_penalty,
            drawdown_penalty=drawdown_penalty,
        )
        observation = environment.reset()
        agent.network.reset()
        scores = agent.observe(observation)
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.choose_action_epsilon_greedy(scores, current_epsilon)
            action_counts[action] += 1
            result = environment.step(action)
            agent.memory.add(observation, action, result.reward, result.observation, result.done)

            if result.done:
                next_scores = (0.0,) * agent.network.output_size
            else:
                next_scores = agent.observe(result.observation)
            agent.network.learn_td(
                action,
                result.reward,
                next_scores,
                result.done,
                agent.learning_rate,
                discount,
                trace_decay,
            )

            total_reward += result.reward
            observation = result.observation
            scores = next_scores
            if result.done:
                break

        metrics = environment.episode_result(total_reward)
        returns.append(metrics.return_pct)
        drawdowns.append(metrics.max_drawdown_pct)
        total_trades += metrics.trades
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return FinanceTrainingResult(
        episodes,
        sum(returns) / len(returns),
        max(returns),
        returns[-1],
        sum(drawdowns) / len(drawdowns),
        total_trades,
        action_counts[0],
        action_counts[1],
        action_counts[2],
    )


def evaluate_synthetic_crypto(
    agent: Agent,
    seed: int = 10_000,
    market_length: int = 256,
    max_steps: int = 200,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
) -> FinanceEvaluationResult:
    """Evaluate greedily on a fresh market without changing agent weights."""
    candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
    environment = CryptoTradingEnvironment(
        candles,
        max_steps=max_steps,
        trade_penalty=trade_penalty,
        invalid_action_penalty=invalid_action_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    observation = environment.reset()
    agent.network.reset()
    scores = agent.observe(observation)
    total_reward = 0.0
    action_counts = [0, 0, 0]

    for _ in range(max_steps):
        action = agent.choose_action(scores)
        action_counts[action] += 1
        result = environment.step(action)
        total_reward += result.reward
        observation = result.observation
        if result.done:
            break
        scores = agent.observe(observation)

    metrics = environment.episode_result(total_reward)
    return FinanceEvaluationResult(
        metrics.return_pct,
        metrics.final_portfolio,
        metrics.max_drawdown_pct,
        metrics.trades,
        action_counts[0],
        action_counts[1],
        action_counts[2],
    )


def evaluate_synthetic_crypto_multi_market(
    agent: Agent,
    seed: int = 10_000,
    markets: int = 20,
    market_length: int = 256,
    max_steps: int = 200,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
) -> FinanceMultiMarketResult:
    """Evaluate the same policy across many unseen markets and baselines."""
    if markets < 1:
        raise ValueError("markets must be >= 1")

    agent_returns: list[float] = []
    drawdowns: list[float] = []
    trades: list[int] = []
    trade_frequencies: list[float] = []
    hold_returns: list[float] = []
    buy_hold_returns: list[float] = []
    wins = 0

    for market_seed in range(seed, seed + markets):
        evaluation = evaluate_synthetic_crypto(
            agent,
            seed=market_seed,
            market_length=market_length,
            max_steps=max_steps,
            trade_penalty=trade_penalty,
            invalid_action_penalty=invalid_action_penalty,
            drawdown_penalty=drawdown_penalty,
        )
        hold = evaluate_hold(seed=market_seed, market_length=market_length, max_steps=max_steps)
        buy_hold = evaluate_buy_and_hold(seed=market_seed, market_length=market_length, max_steps=max_steps)

        agent_returns.append(evaluation.return_pct)
        drawdowns.append(evaluation.max_drawdown_pct)
        trades.append(evaluation.trades)
        trade_frequencies.append(evaluation.trade_frequency)
        hold_returns.append(hold.return_pct)
        buy_hold_returns.append(buy_hold.return_pct)
        wins += evaluation.return_pct > buy_hold.return_pct

    return FinanceMultiMarketResult(
        markets=markets,
        average_return_pct=sum(agent_returns) / markets,
        median_return_pct=median(agent_returns),
        average_drawdown_pct=sum(drawdowns) / markets,
        average_trades=sum(trades) / markets,
        average_trade_frequency=sum(trade_frequencies) / markets,
        win_rate_vs_buy_hold=wins / markets,
        average_excess_return_pct=sum(
            agent_return - buy_hold_return
            for agent_return, buy_hold_return in zip(agent_returns, buy_hold_returns)
        ) / markets,
        hold_average_return_pct=sum(hold_returns) / markets,
        buy_hold_average_return_pct=sum(buy_hold_returns) / markets,
    )
