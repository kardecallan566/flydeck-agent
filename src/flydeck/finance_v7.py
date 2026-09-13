from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .agent import Agent
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder


@dataclass(frozen=True, slots=True)
class FinanceV7Result:
    episodes: int
    average_return_pct: float
    best_return_pct: float
    last_return_pct: float
    average_drawdown_pct: float
    total_trades: int
    raw_hold_actions: int
    raw_buy_actions: int
    raw_sell_actions: int
    executed_hold_actions: int
    executed_buy_actions: int
    executed_sell_actions: int
    gated_actions: int
    average_confidence: float
    median_confidence: float
    average_prediction_error: float
    opportunity_rate: float

    @property
    def raw_total_actions(self) -> int:
        return self.raw_hold_actions + self.raw_buy_actions + self.raw_sell_actions

    @property
    def executed_total_actions(self) -> int:
        return self.executed_hold_actions + self.executed_buy_actions + self.executed_sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.total_trades / max(1, self.executed_total_actions)

    @property
    def gate_activation_rate(self) -> float:
        return self.gated_actions / max(1, self.raw_total_actions)


@dataclass(frozen=True, slots=True)
class FinanceV7Evaluation:
    return_pct: float
    final_portfolio: float
    max_drawdown_pct: float
    trades: int
    raw_hold_actions: int
    raw_buy_actions: int
    raw_sell_actions: int
    executed_hold_actions: int
    executed_buy_actions: int
    executed_sell_actions: int
    gated_actions: int
    average_confidence: float
    median_confidence: float
    average_prediction_error: float
    opportunity_rate: float

    @property
    def raw_total_actions(self) -> int:
        return self.raw_hold_actions + self.raw_buy_actions + self.raw_sell_actions

    @property
    def executed_total_actions(self) -> int:
        return self.executed_hold_actions + self.executed_buy_actions + self.executed_sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.trades / max(1, self.executed_total_actions)

    @property
    def gate_activation_rate(self) -> float:
        return self.gated_actions / max(1, self.raw_total_actions)


@dataclass(frozen=True, slots=True)
class _Gate:
    action: int
    gated: bool
    confidence: float
    adaptive_margin: float
    opportunity: bool


def _softmax(scores: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    scaled = tuple(score / temperature for score in scores)
    maximum = max(scaled)
    exponentials = tuple(math.exp(value - maximum) for value in scaled)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def gate_action(
    scores: tuple[float, ...],
    opportunity_margin: float = 0.08,
    confidence_threshold: float = 0.55,
    temperature: float = 0.02,
) -> _Gate:
    """Turn value evidence into an execution decision without changing learning.

    Confidence is derived from a temperature-scaled softmax rather than from the
    raw score range. This makes the gate sensitive to both the direction of the
    best trade and how decisively it separates from HOLD, while avoiding the V6
    failure mode where one fixed absolute margin dominated tiny recurrent scores.
    """
    if len(scores) != 3:
        raise ValueError("finance scores must contain HOLD, BUY and SELL")
    if opportunity_margin < 0:
        raise ValueError("opportunity_margin must be >= 0")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    best_trade = max(scores[1:])
    best_trade_action = 1 if scores[1] >= scores[2] else 2
    score_range = max(scores) - min(scores)
    trade_advantage = best_trade - scores[0]
    adaptive_margin = max(0.005, min(opportunity_margin, score_range * 0.25))

    probabilities = _softmax(scores, temperature)
    confidence = probabilities[best_trade_action] if trade_advantage > 0 else 0.0
    opportunity = trade_advantage > adaptive_margin and confidence >= confidence_threshold

    if not opportunity:
        return _Gate(0, True, confidence, adaptive_margin, False)
    return _Gate(best_trade_action, False, confidence, adaptive_margin, True)


def _make_encoder(agent: Agent) -> SparseMarketEncoder:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("V7 finance agent requires a 27-input / 3-action sparse network")
    return encoder


def _observe(agent: Agent, encoder: SparseMarketEncoder, observation: tuple[float, ...]) -> tuple[float, ...]:
    return agent.observe(encoder.encode(observation))


def _run_episode(
    agent: Agent,
    candles: tuple,
    max_steps: int,
    epsilon: float,
    opportunity_margin: float,
    confidence_threshold: float,
    trade_penalty: float,
    invalid_action_penalty: float,
    drawdown_penalty: float,
    discount: float,
    trace_decay: float,
    train: bool,
) -> tuple[float, float, int, list[int], list[int], int, list[float], list[float], int]:
    environment = CryptoTradingEnvironment(
        candles,
        max_steps=max_steps,
        trade_penalty=trade_penalty,
        invalid_action_penalty=invalid_action_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    encoder = _make_encoder(agent)
    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = _observe(agent, encoder, observation)

    raw_counts = [0, 0, 0]
    executed_counts = [0, 0, 0]
    gated_actions = 0
    confidences: list[float] = []
    prediction_errors: list[float] = []
    opportunities = 0
    total_reward = 0.0

    for _ in range(max_steps):
        gate = gate_action(scores, opportunity_margin, confidence_threshold)
        raw_action = agent.choose_action(scores)
        if train and agent._rng.random() < epsilon:
            raw_action = agent._rng.randrange(3)

        raw_counts[raw_action] += 1
        confidences.append(gate.confidence)
        opportunities += int(gate.opportunity)

        # Training must expose the raw policy to the real environment. Applying
        # the gate during learning would again create a credit-assignment error:
        # the network chooses BUY, the gate executes HOLD, and TD would attribute
        # HOLD's reward to BUY. The gate is therefore strictly an execution layer.
        if train:
            executed_action = raw_action
            gated = False
        elif raw_action == 0:
            executed_action = 0
            gated = False
        elif gate.gated:
            executed_action = 0
            gated = True
        else:
            executed_action = raw_action
            gated = False

        if gated:
            gated_actions += 1
        executed_counts[executed_action] += 1

        result = environment.step(executed_action)
        encoder.observe_action(executed_action)

        if result.done:
            next_scores = (0.0, 0.0, 0.0)
        else:
            next_scores = _observe(agent, encoder, result.observation)

        if train:
            td_error = agent.network.learn_td(
                raw_action,
                result.reward,
                next_scores,
                result.done,
                agent.learning_rate,
                discount,
                trace_decay,
            )
            prediction_errors.append(abs(td_error))

        total_reward += result.reward
        observation = result.observation
        scores = next_scores
        if result.done:
            break

    metrics = environment.episode_result(total_reward)
    return (
        metrics.return_pct,
        metrics.max_drawdown_pct,
        metrics.trades,
        raw_counts,
        executed_counts,
        gated_actions,
        confidences,
        prediction_errors,
        opportunities,
    )


def train_synthetic_crypto_v7(
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
    opportunity_margin: float = 0.08,
    confidence_threshold: float = 0.55,
) -> FinanceV7Result:
    if episodes < 1 or market_length < 32 or max_steps < 1:
        raise ValueError("episodes >= 1, market_length >= 32 and max_steps >= 1 are required")

    returns: list[float] = []
    drawdowns: list[float] = []
    raw_total = [0, 0, 0]
    executed_total = [0, 0, 0]
    total_trades = 0
    gated = 0
    confidences: list[float] = []
    prediction_errors: list[float] = []
    opportunities = 0
    actions = 0
    current_epsilon = epsilon

    for episode in range(episodes):
        result = _run_episode(
            agent,
            SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate(),
            max_steps,
            current_epsilon,
            opportunity_margin,
            confidence_threshold,
            trade_penalty,
            invalid_action_penalty,
            drawdown_penalty,
            discount,
            trace_decay,
            True,
        )
        return_pct, drawdown_pct, trades, raw_counts, executed_counts, gated_count, confs, errors, opps = result
        returns.append(return_pct)
        drawdowns.append(drawdown_pct)
        total_trades += trades
        gated += gated_count
        opportunities += opps
        actions += sum(raw_counts)
        for index in range(3):
            raw_total[index] += raw_counts[index]
            executed_total[index] += executed_counts[index]
        confidences.extend(confs)
        prediction_errors.extend(errors)
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return FinanceV7Result(
        episodes=episodes,
        average_return_pct=sum(returns) / len(returns),
        best_return_pct=max(returns),
        last_return_pct=returns[-1],
        average_drawdown_pct=sum(drawdowns) / len(drawdowns),
        total_trades=total_trades,
        raw_hold_actions=raw_total[0],
        raw_buy_actions=raw_total[1],
        raw_sell_actions=raw_total[2],
        executed_hold_actions=executed_total[0],
        executed_buy_actions=executed_total[1],
        executed_sell_actions=executed_total[2],
        gated_actions=gated,
        average_confidence=sum(confidences) / max(1, len(confidences)),
        median_confidence=median(confidences) if confidences else 0.0,
        average_prediction_error=sum(prediction_errors) / max(1, len(prediction_errors)),
        opportunity_rate=opportunities / max(1, actions),
    )


def evaluate_synthetic_crypto_v7(
    agent: Agent,
    seed: int = 10_000,
    market_length: int = 256,
    max_steps: int = 200,
    trade_penalty: float = 0.0025,
    invalid_action_penalty: float = 0.001,
    drawdown_penalty: float = 0.02,
    opportunity_margin: float = 0.08,
    confidence_threshold: float = 0.55,
) -> FinanceV7Evaluation:
    result = _run_episode(
        agent,
        SyntheticCryptoMarket(length=market_length, seed=seed).generate(),
        max_steps,
        0.0,
        opportunity_margin,
        confidence_threshold,
        trade_penalty,
        invalid_action_penalty,
        drawdown_penalty,
        0.97,
        0.85,
        False,
    )
    return_pct, drawdown_pct, trades, raw, executed, gated, confs, errors, opportunities = result
    return FinanceV7Evaluation(
        return_pct=return_pct,
        final_portfolio=10_000.0 * (1.0 + return_pct / 100.0),
        max_drawdown_pct=drawdown_pct,
        trades=trades,
        raw_hold_actions=raw[0],
        raw_buy_actions=raw[1],
        raw_sell_actions=raw[2],
        executed_hold_actions=executed[0],
        executed_buy_actions=executed[1],
        executed_sell_actions=executed[2],
        gated_actions=gated,
        average_confidence=sum(confs) / max(1, len(confs)),
        median_confidence=median(confs) if confs else 0.0,
        average_prediction_error=sum(errors) / max(1, len(errors)),
        opportunity_rate=opportunities / max(1, len(confs)),
    )
