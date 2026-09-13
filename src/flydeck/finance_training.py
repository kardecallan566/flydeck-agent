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

    returns: list[float] = []
    drawdowns: list[float] = []
    total_trades = 0
    current_epsilon = epsilon

    for episode in range(episodes):
        candles = SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate()
        environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
        observation = environment.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.act_epsilon_greedy(observation, current_epsilon)
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
    )
