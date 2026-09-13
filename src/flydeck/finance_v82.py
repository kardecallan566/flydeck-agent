from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from statistics import mean

from .agent import Agent
from .finance import Candle, CryptoTradingEnvironment
from .finance_encoder import SparseMarketEncoder
from .finance_real import HORIZONS


@dataclass(frozen=True, slots=True)
class PositionDecisionRecord:
    index: int
    action: int
    position_before: float
    position_after: float
    position_state_before: str
    position_state_after: str
    exposure_change: float
    portfolio_value_before: float
    portfolio_value_after: float
    scores: tuple[float, float, float]
    counterfactual_rewards: tuple[float, float, float]
    realized_portfolio_returns: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class PositionAwareDiagnostics:
    records: tuple[PositionDecisionRecord, ...]
    action_state_transitions: tuple[tuple[int, ...], ...]
    position_state_transitions: tuple[tuple[int, ...], ...]
    average_exposure_change_by_action: tuple[float, float, float]
    average_counterfactual_rewards: tuple[float, float, float]
    average_realized_portfolio_returns_by_action: tuple[tuple[float, ...], ...]
    average_realized_portfolio_returns_by_position_state: tuple[tuple[float, ...], ...]
    average_buy_advantage_by_position_state: tuple[float, float]
    average_sell_advantage_by_position_state: tuple[float, float]

    @property
    def flat_records(self) -> int:
        return sum(1 for record in self.records if record.position_state_before == "FLAT")

    @property
    def long_records(self) -> int:
        return sum(1 for record in self.records if record.position_state_before == "LONG")


def _position_state(position_ratio: float) -> str:
    return "LONG" if position_ratio > 1e-12 else "FLAT"


def _portfolio_returns(
    portfolio_values: dict[int, float],
    index: int,
    current_value: float,
    horizons: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    if current_value <= 0:
        return ()
    return tuple(
        (horizon, portfolio_values[index + horizon] / current_value - 1.0)
        for horizon in horizons
        if index + horizon in portfolio_values
    )


def collect_position_aware_diagnostics(
    agent: Agent,
    candles: tuple[Candle, ...],
    *,
    max_steps: int | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> PositionAwareDiagnostics:
    """Audit V8 decisions against portfolio state and exact immediate counterfactuals.

    Diagnostic-only: learned weights are never updated. Each counterfactual is
    evaluated from a deep copy of the exact pre-action environment state.
    """
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("real finance benchmark requires a 27-input / 3-action agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    records: list[PositionDecisionRecord] = []
    action_state_transitions = [[0, 0, 0] for _ in range(3)]
    position_state_transitions = [[0, 0] for _ in range(2)]
    portfolio_values: dict[int, float] = {environment.index: environment.portfolio_value}
    exposure_changes = [[] for _ in range(3)]
    counterfactual_rewards = [[] for _ in range(3)]
    previous_action: int | None = None

    for _ in range(environment.max_steps):
        index = environment.index
        action = agent.choose_action(scores)
        position_before = environment.position_ratio
        state_before = _position_state(position_before)
        portfolio_before = environment.portfolio_value

        if previous_action is not None:
            action_state_transitions[previous_action][action] += 1
        previous_action = action

        counterfactuals: list[float] = []
        for counterfactual_action in range(3):
            trial = deepcopy(environment)
            reward = trial.step(counterfactual_action).reward
            counterfactuals.append(reward)
            counterfactual_rewards[counterfactual_action].append(reward)

        result = environment.step(action)
        position_after = environment.position_ratio
        state_after = _position_state(position_after)
        portfolio_after = environment.portfolio_value
        exposure_change = position_after - position_before
        exposure_changes[action].append(exposure_change)
        position_state_transitions[0 if state_before == "FLAT" else 1][0 if state_after == "FLAT" else 1] += 1
        portfolio_values[environment.index] = portfolio_after

        records.append(
            PositionDecisionRecord(
                index=index,
                action=action,
                position_before=position_before,
                position_after=position_after,
                position_state_before=state_before,
                position_state_after=state_after,
                exposure_change=exposure_change,
                portfolio_value_before=portfolio_before,
                portfolio_value_after=portfolio_after,
                scores=tuple(scores),
                counterfactual_rewards=tuple(counterfactuals),
                realized_portfolio_returns=(),
            )
        )

        encoder.observe_action(action)
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    completed_records: list[PositionDecisionRecord] = []
    action_returns = [[[] for _ in horizons] for _ in range(3)]
    state_returns = [[[] for _ in horizons] for _ in range(2)]
    buy_advantages = [[], []]
    sell_advantages = [[], []]

    for record in records:
        realized = _portfolio_returns(portfolio_values, record.index, record.portfolio_value_before, horizons)
        state_index = 0 if record.position_state_before == "FLAT" else 1
        for horizon, value in realized:
            horizon_index = horizons.index(horizon)
            action_returns[record.action][horizon_index].append(value)
            state_returns[state_index][horizon_index].append(value)
        buy_advantages[state_index].append(record.scores[1] - record.scores[0])
        sell_advantages[state_index].append(record.scores[2] - record.scores[0])
        completed_records.append(
            PositionDecisionRecord(
                index=record.index,
                action=record.action,
                position_before=record.position_before,
                position_after=record.position_after,
                position_state_before=record.position_state_before,
                position_state_after=record.position_state_after,
                exposure_change=record.exposure_change,
                portfolio_value_before=record.portfolio_value_before,
                portfolio_value_after=record.portfolio_value_after,
                scores=record.scores,
                counterfactual_rewards=record.counterfactual_rewards,
                realized_portfolio_returns=realized,
            )
        )

    average_action_returns = tuple(
        tuple(mean(values) if values else 0.0 for values in action_horizons)
        for action_horizons in action_returns
    )
    average_state_returns = tuple(
        tuple(mean(values) if values else 0.0 for values in state_horizons)
        for state_horizons in state_returns
    )
    return PositionAwareDiagnostics(
        records=tuple(completed_records),
        action_state_transitions=tuple(tuple(row) for row in action_state_transitions),
        position_state_transitions=tuple(tuple(row) for row in position_state_transitions),
        average_exposure_change_by_action=tuple(mean(values) if values else 0.0 for values in exposure_changes),
        average_counterfactual_rewards=tuple(mean(values) if values else 0.0 for values in counterfactual_rewards),
        average_realized_portfolio_returns_by_action=average_action_returns,
        average_realized_portfolio_returns_by_position_state=average_state_returns,
        average_buy_advantage_by_position_state=tuple(mean(values) if values else 0.0 for values in buy_advantages),
        average_sell_advantage_by_position_state=tuple(mean(values) if values else 0.0 for values in sell_advantages),
    )
