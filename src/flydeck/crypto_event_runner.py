from __future__ import annotations

from dataclasses import dataclass
import math

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .crypto_event_policy import CryptoEventConfig, CryptoEventLabeler, CryptoEventPolicy
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class CryptoEventMetrics:
    rounds: int
    signals: int
    positive_signals: int
    negative_signals: int
    flat_signals: int
    total_return: float
    max_drawdown: float
    volatility: float
    sharpe_like: float
    hit_rate: float
    average_position: float
    average_horizon: float
    labels_up: int
    labels_down: int
    labels_wait: int


def run_crypto_event_benchmark(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    *,
    context: int = 32,
    config: CryptoEventConfig | None = None,
) -> tuple[CryptoEventMetrics, CryptoEventMetrics, CryptoEventMetrics]:
    if context < 4:
        raise ValueError("context must be at least four candles")
    config = config or CryptoEventConfig()
    usable = data.size - max(config.horizons)
    if usable < context + 20:
        raise ValueError("dataset is too small for crypto-event benchmark")
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    agent = FlyVisualPredictionAgent(circuit, retina_width=context)
    policy = CryptoEventPolicy(config)
    labeler = CryptoEventLabeler(config)
    train = _split(agent, policy, labeler, data, context - 1, train_end, context, learn=True)
    agent.set_learning(False)
    agent.reset(preserve_learning=True)
    validation = _split(agent, policy, labeler, data, train_end, validation_end, context, learn=False)
    agent.reset(preserve_learning=True)
    test = _split(agent, policy, labeler, data, validation_end, usable, context, learn=False)
    return train, validation, test


def _split(agent: FlyVisualPredictionAgent, policy: CryptoEventPolicy, labeler: CryptoEventLabeler,
           data: BNBPredictionDataset, start: int, end: int, context: int, learn: bool) -> CryptoEventMetrics:
    pnl: list[float] = []
    equity = 1.0
    peak = equity
    max_drawdown = 0.0
    signals = positive = negative = flat = hits = 0
    positions: list[float] = []
    horizons: list[int] = []
    labels = {Prediction.UP: 0, Prediction.DOWN: 0, Prediction.WAIT: 0}
    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1): index + 1]
        volumes = data.volumes[max(0, index - context + 1): index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=volumes)
        state = agent.internal_state
        action = policy.decide(decision, fast_memory=state.fast_memory, slow_memory=state.slow_memory,
                               uncertainty=state.uncertainty, novelty=state.feature_novelty, regime=decision.regime)
        target = labeler.target(data, index)
        label_index = labeler.config.horizons.index(action.horizon)
        label = target.labels[label_index]
        labels[label] += 1
        if learn:
            # The visual agent's own reward remains strictly one-step causal.
            pass
        realized = data.closes[index + action.horizon] / data.closes[index] - 1.0
        trade_pnl = action.position * realized - abs(action.position) * config_cost(labeler.config)
        equity *= 1.0 + trade_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / max(1e-12, peak))
        pnl.append(trade_pnl)
        positions.append(abs(action.position))
        horizons.append(action.horizon)
        if abs(action.position) < 0.05:
            flat += 1
        elif action.position > 0.0:
            positive += 1
            signals += 1
            hits += int(realized > labeler.config.round_trip_cost)
        else:
            negative += 1
            signals += 1
            hits += int(realized < -labeler.config.round_trip_cost)
    mean = sum(pnl) / len(pnl) if pnl else 0.0
    variance = sum((value - mean) ** 2 for value in pnl) / len(pnl) if pnl else 0.0
    volatility = math.sqrt(max(0.0, variance))
    return CryptoEventMetrics(
        rounds=end - start, signals=signals, positive_signals=positive, negative_signals=negative,
        flat_signals=flat, total_return=equity - 1.0, max_drawdown=max_drawdown,
        volatility=volatility, sharpe_like=mean / volatility * math.sqrt(288.0) if volatility > 1e-12 else 0.0,
        hit_rate=hits / signals if signals else 0.0,
        average_position=sum(positions) / len(positions) if positions else 0.0,
        average_horizon=sum(horizons) / len(horizons) if horizons else 0.0,
        labels_up=labels[Prediction.UP], labels_down=labels[Prediction.DOWN], labels_wait=labels[Prediction.WAIT],
    )


def config_cost(config: CryptoEventConfig) -> float:
    return config.round_trip_cost
