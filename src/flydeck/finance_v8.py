from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder


@dataclass(frozen=True, slots=True)
class FinanceV8Result:
    episodes: int
    average_return_pct: float
    best_return_pct: float
    last_return_pct: float
    average_drawdown_pct: float
    total_trades: int
    hold_actions: int
    buy_actions: int
    sell_actions: int
    average_prediction_error: float

    @property
    def total_actions(self) -> int:
        return self.hold_actions + self.buy_actions + self.sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.total_trades / max(1, self.total_actions)


def _run_episode(
    agent: Agent,
    candles: tuple,
    max_steps: int,
    epsilon: float,
    trade_penalty: float,
    invalid_action_penalty: float,
    drawdown_penalty: float,
    discount: float,
    trace_decay: float,
) -> tuple[float, float, int, list[int], list[float]]:
    environment = CryptoTradingEnvironment(
        candles,
        max_steps=max_steps,
        trade_penalty=trade_penalty,
        invalid_action_penalty=invalid_action_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("V8 finance agent requires a 27-input / 3-action sparse network")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    counts = [0, 0, 0]
    errors: list[float] = []
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent.choose_action(scores)
        if agent._rng.random() < epsilon:
            action = agent._rng.randrange(3)
        counts[action] += 1

        result = environment.step(action)
        encoder.observe_action(action)
        if result.done:
            next_scores = (0.0, 0.0, 0.0)
        else:
            next_scores = agent.observe(encoder.encode(result.observation))

        td_error = agent.network.learn_td_selected_action(
            action,
            result.reward,
            next_scores,
            result.done,
            agent.learning_rate,
            discount,
            trace_decay,
        )
        errors.append(abs(td_error))
        total_reward += result.reward
        scores = next_scores
        if result.done:
            break

    metrics = environment.episode_result(total_reward)
    return metrics.return_pct, metrics.max_drawdown_pct, metrics.trades, counts, errors


def train_synthetic_crypto_v8(
    agent: Agent,
    episodes: int = 100,
    market_length: int = 256,
    max_steps: int = 200,
    seed: int = 42,
    epsilon: float = 0.30,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.05,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
    discount: float = 0.97,
    trace_decay: float = 0.85,
) -> FinanceV8Result:
    if episodes < 1 or market_length < 32 or max_steps < 1:
        raise ValueError("episodes >= 1, market_length >= 32 and max_steps >= 1 are required")

    returns: list[float] = []
    drawdowns: list[float] = []
    errors: list[float] = []
    counts = [0, 0, 0]
    total_trades = 0
    current_epsilon = epsilon

    for episode in range(episodes):
        result = _run_episode(
            agent,
            SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate(),
            max_steps,
            current_epsilon,
            trade_penalty,
            invalid_action_penalty,
            drawdown_penalty,
            discount,
            trace_decay,
        )
        return_pct, drawdown_pct, trades, episode_counts, episode_errors = result
        returns.append(return_pct)
        drawdowns.append(drawdown_pct)
        errors.extend(episode_errors)
        total_trades += trades
        for index in range(3):
            counts[index] += episode_counts[index]
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return FinanceV8Result(
        episodes=episodes,
        average_return_pct=sum(returns) / len(returns),
        best_return_pct=max(returns),
        last_return_pct=returns[-1],
        average_drawdown_pct=sum(drawdowns) / len(drawdowns),
        total_trades=total_trades,
        hold_actions=counts[0],
        buy_actions=counts[1],
        sell_actions=counts[2],
        average_prediction_error=sum(errors) / max(1, len(errors)),
    )
