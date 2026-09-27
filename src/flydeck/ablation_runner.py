from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .crypto_event_policy import CryptoEventConfig
from .crypto_event_runner import CryptoEventMetrics, run_crypto_event_benchmark
from .economic_metrics import EconomicSurvivalMetrics, calculate_economic_survival_metrics
from .risk_policy import LightweightRiskPolicy
from .standard_metrics import StandardizedMetrics, standardize_metrics
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class AblationResult:
    variant: str
    split: str
    standardized: StandardizedMetrics
    economic: EconomicSurvivalMetrics


def run_ablation_matrix(data: BNBPredictionDataset, circuit: VisualCircuit, *, context: int = 32,
                        config: CryptoEventConfig | None = None) -> tuple[AblationResult, ...]:
    config = config or CryptoEventConfig()
    usable = data.size - max(config.horizons)
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    rows: list[AblationResult] = []
    for variant, risk in (("baseline_original", False), ("malecns_risk", True)):
        agent = FlyVisualPredictionAgent(circuit, retina_width=context)
        risk_policy = LightweightRiskPolicy() if risk else None
        rows.extend(_run_simple_variant(variant, agent, risk_policy, data, context - 1, train_end, context, "TRAIN", config))
        agent.set_learning(False)
        agent.reset(preserve_learning=True)
        if risk_policy:
            risk_policy.reset()
        rows.extend(_run_simple_variant(variant, agent, risk_policy, data, train_end, validation_end, context, "VALIDATION", config))
        agent.reset(preserve_learning=True)
        if risk_policy:
            risk_policy.reset()
        rows.extend(_run_simple_variant(variant, agent, risk_policy, data, validation_end, usable, context, "TEST", config))
    current = run_crypto_event_benchmark(data, circuit, context=context, config=config)
    for split, metrics in zip(("TRAIN", "VALIDATION", "TEST"), current):
        rows.append(_from_current(split, metrics))
    return tuple(rows)


def _run_simple_variant(variant: str, agent: FlyVisualPredictionAgent, risk: LightweightRiskPolicy | None,
                        data: BNBPredictionDataset, start: int, end: int, context: int, split: str,
                        config: CryptoEventConfig) -> tuple[AblationResult, ...]:
    correct = profitable = entered = 0
    returns: list[float] = []
    positions: list[float] = []
    costs: list[float] = []
    previous = 0.0
    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1): index + 1]
        volumes = data.volumes[max(0, index - context + 1): index + 1]
        _, decision = agent.perceive(prices, volumes=volumes)
        action = risk.before_action(decision.action, novelty=agent.internal_state.feature_novelty) if risk else decision.action
        outcome = data.outcome(index)
        direction = 1 if action == Prediction.UP else -1 if action == Prediction.DOWN else 0
        if direction:
            entered += 1
            correct += int((direction > 0 and outcome == Prediction.UP) or (direction < 0 and outcome == Prediction.DOWN))
        position = 0.10 * direction
        realized = data.closes[index + 1] / data.closes[index] - 1.0
        cost = abs(position - previous) * config.round_trip_cost
        net = position * realized - cost
        returns.append(net)
        positions.append(position)
        costs.append(cost)
        profitable += int(direction != 0 and net > 0.0)
        previous = position
        if risk:
            risk.observe(action, net)
    standardized = standardize_metrics(rounds=end - start, entered=entered, correct=correct,
                                        profitable=profitable, net_returns=returns)
    economic = calculate_economic_survival_metrics(returns, positions, costs=costs)
    return (AblationResult(variant, split, standardized, economic),)


def _from_current(split: str, metrics: CryptoEventMetrics) -> AblationResult:
    return AblationResult(
        variant="current_temporal_crypto_event", split=split,
        standardized=StandardizedMetrics(
            rounds=metrics.rounds, entered=metrics.signals, correct=round(metrics.accuracy * metrics.signals),
            profitable=round(metrics.hit_rate * metrics.signals), accuracy=metrics.accuracy,
            hit_rate=metrics.hit_rate, coverage=metrics.coverage, economic_return=metrics.total_return),
        economic=metrics.economic,
    )
