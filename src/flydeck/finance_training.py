from __future__ import annotations

from dataclasses import dataclass

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


def train_synthetic_crypto(
    agent: Agent,
    episodes: int = 100,
    market_length: int = 256,
    max_steps: int = 200,
    seed: int = 42,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.03,
) -> FinanceTrainingResult:
    """Train across a different generated market on every episode."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if market_length < 32:
        raise ValueError("market_length must be >= 32")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1]")
    if not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("min_epsilon must be between 0 and 1")

    returns: list[float] = []
    drawdowns: list[float] = []
    total_trades = 0
    action_counts = [0, 0, 0]
    current_epsilon = epsilon

    for episode in range(episodes):
        candles = SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate()
        environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
        observation = environment.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.act_epsilon_greedy(observation, current_epsilon)
            action_counts[action] += 1
            result = environment.step(action)
            agent.memory.add(observation, action, result.reward, result.observation, result.done)
            agent.network.learn(action, result.reward, agent.learning_rate)
            total_reward += result.reward
            observation = result.observation
            if result.done:
                break

        metrics = environment.episode_result(total_reward)
        returns.append(metrics.return_pct)
        drawdowns.append(metrics.max_drawdown_pct)
        total_trades += metrics.trades
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return FinanceTrainingResult(
        episodes=episodes,
        average_return_pct=sum(returns) / len(returns),
        best_return_pct=max(returns),
        last_return_pct=returns[-1],
        average_drawdown_pct=sum(drawdowns) / len(drawdowns),
        total_trades=total_trades,
        hold_actions=action_counts[0],
        buy_actions=action_counts[1],
        sell_actions=action_counts[2],
    )


def evaluate_synthetic_crypto(
    agent: Agent,
    seed: int = 10_000,
    market_length: int = 256,
    max_steps: int = 200,
) -> FinanceEvaluationResult:
    """Evaluate greedily on a fresh market without changing agent weights."""
    candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    observation = environment.reset()
    total_reward = 0.0
    action_counts = [0, 0, 0]

    for _ in range(max_steps):
        action = agent.act(observation)
        action_counts[action] += 1
        result = environment.step(action)
        total_reward += result.reward
        observation = result.observation
        if result.done:
            break

    metrics = environment.episode_result(total_reward)
    return FinanceEvaluationResult(
        return_pct=metrics.return_pct,
        final_portfolio=metrics.final_portfolio,
        max_drawdown_pct=metrics.max_drawdown_pct,
        trades=metrics.trades,
        hold_actions=action_counts[0],
        buy_actions=action_counts[1],
        sell_actions=action_counts[2],
    )
