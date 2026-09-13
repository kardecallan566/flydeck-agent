from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

from .agent import Agent
from .finance import Candle, CryptoTradingEnvironment
from .finance_encoder import SparseMarketEncoder


@dataclass(frozen=True, slots=True)
class RegimeStats:
    hold: int
    buy: int
    sell: int

    @property
    def total(self) -> int:
        return self.hold + self.buy + self.sell


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    label: str
    actions: tuple[int, int, int]
    average_scores: tuple[float, float, float]
    average_score_spread: float
    unique_sparse_states: int
    average_active_magnitude: float
    inferred_regimes: dict[str, RegimeStats]

    @property
    def total_actions(self) -> int:
        return sum(self.actions)


def _regime(observation: tuple[float, ...]) -> str:
    """Infer a coarse regime from the same normalized market features the agent sees.

    This is diagnostic only. SyntheticCryptoMarket does not expose regime labels in
    Candle, so the classifier deliberately uses observable features instead of
    reconstructing the generator's hidden state.
    """
    short_return = observation[1]
    volatility = observation[6]
    range_position = observation[8]

    if volatility >= 0.55:
        return "volatile"
    if short_return >= 0.12 and range_position >= 0.55:
        return "trend_up"
    if short_return <= -0.12 and range_position <= 0.45:
        return "trend_down"
    if short_return * observation[3] < -0.02:
        return "reversal"
    return "sideways"


def diagnose_policy(
    agent: Agent,
    candles: tuple[Candle, ...] | list[Candle],
    *,
    label: str,
    max_steps: int = 200,
) -> PolicyDiagnostics:
    """Inspect the trained policy without changing weights or applying the gate."""
    environment = CryptoTradingEnvironment(candles, max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    if agent.network.input_size != encoder.output_size or agent.network.output_size != 3:
        raise ValueError("diagnostics require a 27-input / 3-action finance agent")

    observation = environment.reset()
    agent.network.reset()
    encoder.reset()

    actions = [0, 0, 0]
    score_sums = [0.0, 0.0, 0.0]
    spreads: list[float] = []
    active_magnitudes: list[float] = []
    sparse_states: set[tuple[float, ...]] = set()
    regime_counts = {name: [0, 0, 0] for name in ("trend_up", "trend_down", "sideways", "volatile", "reversal")}

    for _ in range(max_steps):
        encoded = encoder.encode(observation)
        sparse_states.add(encoded)
        active_magnitudes.append(sum(abs(value) for value in encoded))
        scores = agent.observe(encoded)
        action = agent.choose_action(scores)
        actions[action] += 1
        for index, score in enumerate(scores):
            score_sums[index] += score
        spreads.append(max(scores) - min(scores))
        regime_counts[_regime(observation)][action] += 1

        result = environment.step(action)
        encoder.observe_action(action)
        observation = result.observation
        if result.done:
            break

    count = max(1, sum(actions))
    regimes = {
        name: RegimeStats(*counts) for name, counts in regime_counts.items()
    }
    return PolicyDiagnostics(
        label=label,
        actions=tuple(actions),
        average_scores=tuple(score / count for score in score_sums),
        average_score_spread=sum(spreads) / max(1, len(spreads)),
        unique_sparse_states=len(sparse_states),
        average_active_magnitude=sum(active_magnitudes) / max(1, len(active_magnitudes)),
        inferred_regimes=regimes,
    )


def format_diagnostics(diagnostics: PolicyDiagnostics) -> str:
    lines = [
        f"{diagnostics.label}",
        "actions:",
        f"  HOLD: {diagnostics.actions[0]:4d} ({diagnostics.actions[0] / max(1, diagnostics.total_actions):.1%})",
        f"  BUY:  {diagnostics.actions[1]:4d} ({diagnostics.actions[1] / max(1, diagnostics.total_actions):.1%})",
        f"  SELL: {diagnostics.actions[2]:4d} ({diagnostics.actions[2] / max(1, diagnostics.total_actions):.1%})",
        "average scores:",
        f"  HOLD: {diagnostics.average_scores[0]: .6f}",
        f"  BUY:  {diagnostics.average_scores[1]: .6f}",
        f"  SELL: {diagnostics.average_scores[2]: .6f}",
        f"average score spread: {diagnostics.average_score_spread:.6f}",
        f"unique sparse states: {diagnostics.unique_sparse_states}",
        f"average encoded magnitude: {diagnostics.average_active_magnitude:.6f}",
        "inferred regimes (action counts):",
        "  regime       HOLD  BUY  SELL",
    ]
    for name, stats in diagnostics.inferred_regimes.items():
        lines.append(f"  {name:<12} {stats.hold:4d} {stats.buy:4d} {stats.sell:5d}")
    return "\n".join(lines)
