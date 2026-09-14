from __future__ import annotations

from copy import copy
from dataclasses import dataclass

from .agent import Agent
from .finance import Candle, CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder


@dataclass(frozen=True, slots=True)
class FinanceV83Result:
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


def _counterfactual_targets(
    agent: Agent,
    encoder: SparseMarketEncoder,
    environment: CryptoTradingEnvironment,
    *,
    discount: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Evaluate every action without deep-copying the full simulation graph.

    The environment's market data is immutable, so a shallow copy is enough to
    isolate its mutable episode state. The encoder only has one mutable scalar,
    and the network's step state is isolated by copying its small state vectors;
    its immutable Connection objects can be safely shared. This preserves the
    exact causal successor calculation while avoiding three full deep copies per
    decision.
    """
    rewards: list[float] = []
    next_scores: list[tuple[float, ...]] = []
    dones: list[bool] = []

    for action in range(environment.action_size):
        trial = copy(environment)
        trial_encoder = copy(encoder)
        trial_network = copy(agent.network)

        # SparseNetwork.step mutates only these state vectors. Connections are
        # frozen dataclasses and therefore safe to share between shallow copies.
        trial_network._state = list(agent.network._state)
        trial_network._decision_state = list(agent.network._decision_state)
        trial_network._previous_decision_state = list(agent.network._previous_decision_state)
        trial_network._eligibility = list(agent.network._eligibility)

        result = trial.step(action)
        rewards.append(result.reward)
        dones.append(result.done)
        if result.done:
            next_scores.append((0.0,) * agent.network.output_size)
        else:
            trial_encoder.observe_action(action)
            next_scores.append(trial_network.step(trial_encoder.encode(result.observation)))

    return tuple(rewards), tuple(next_scores), tuple(dones)


def _run_episode(
    agent: Agent,
    candles: tuple[Candle, ...],
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
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != environment.action_size:
        raise ValueError("V8.3 finance agent requires a 27-input / 3-action sparse network")

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
            action = agent._rng.randrange(environment.action_size)
        counts[action] += 1

        rewards, next_scores_by_action, done_by_action = _counterfactual_targets(
            agent,
            encoder,
            environment,
            discount=discount,
        )
        td_errors = agent.network.learn_td_all_actions(
            rewards,
            next_scores_by_action,
            done_by_action,
            agent.learning_rate,
            discount,
            trace_decay,
        )
        errors.extend(abs(error) for error in td_errors)

        result = environment.step(action)
        encoder.observe_action(action)
        total_reward += result.reward
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    metrics = environment.episode_result(total_reward)
    return metrics.return_pct, metrics.max_drawdown_pct, metrics.trades, counts, errors


def train_synthetic_crypto_v83(
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
) -> FinanceV83Result:
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

    return FinanceV83Result(
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


def train_real_market_v83(
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
) -> tuple[float, float, int, int, tuple[int, int, int], float]:
    """Train V8.3 through one chronological real-market pass."""
    environment = CryptoTradingEnvironment(
        candles,
        max_steps=max_steps,
        trade_penalty=trade_penalty,
        invalid_action_penalty=invalid_action_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != environment.action_size:
        raise ValueError("V8.3 real finance benchmark requires a 27-input / 3-action agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    counts = [0, 0, 0]
    total_reward = 0.0
    current_epsilon = epsilon
    errors: list[float] = []

    for _ in range(environment.max_steps):
        action = agent.choose_action(scores)
        if agent._rng.random() < current_epsilon:
            action = agent._rng.randrange(environment.action_size)
        counts[action] += 1

        rewards, next_scores_by_action, done_by_action = _counterfactual_targets(
            agent,
            encoder,
            environment,
            discount=discount,
        )
        td_errors = agent.network.learn_td_all_actions(
            rewards,
            next_scores_by_action,
            done_by_action,
            agent.learning_rate,
            discount,
            trace_decay,
        )
        errors.extend(abs(error) for error in td_errors)

        result = environment.step(action)
        encoder.observe_action(action)
        total_reward += result.reward
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    metrics = environment.episode_result(total_reward)
    return (
        metrics.return_pct,
        metrics.max_drawdown_pct,
        metrics.trades,
        metrics.steps,
        tuple(counts),
        sum(errors) / max(1, len(errors)),
    )
