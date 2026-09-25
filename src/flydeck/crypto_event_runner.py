from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .crypto_event_policy import CryptoEventConfig, CryptoEventLabeler, CryptoEventPolicy
from .economic_metrics import EconomicSurvivalMetrics, calculate_economic_survival_metrics
from .temporal_events import TemporalEventExtractor
from .temporal_memory import SparseTemporalMemory
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
    economic: EconomicSurvivalMetrics
    temporal_matches: int
    temporal_attention: float


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
    extractor = TemporalEventExtractor()
    memory = SparseTemporalMemory()
    train = _split(agent, policy, labeler, extractor, memory, data, context - 1, train_end, context, learn=True)
    agent.set_learning(False)
    agent.reset(preserve_learning=True)
    validation = _split(agent, policy, labeler, extractor, memory, data, train_end, validation_end, context, learn=False)
    agent.reset(preserve_learning=True)
    test = _split(agent, policy, labeler, extractor, memory, data, validation_end, usable, context, learn=False)
    return train, validation, test


def _split(agent: FlyVisualPredictionAgent, policy: CryptoEventPolicy, labeler: CryptoEventLabeler,
           extractor: TemporalEventExtractor, memory: SparseTemporalMemory,
           data: BNBPredictionDataset, start: int, end: int, context: int, learn: bool) -> CryptoEventMetrics:
    returns: list[float] = []
    costs: list[float] = []
    positions: list[float] = []
    horizons: list[int] = []
    labels = {Prediction.UP: 0, Prediction.DOWN: 0, Prediction.WAIT: 0}
    signals = positive = negative = flat = hits = matches = 0
    attention_total = 0.0
    previous_position = 0.0
    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1): index + 1]
        volumes = data.volumes[max(0, index - context + 1): index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=volumes)
        state = agent.internal_state
        event = extractor.extract(timestamp=index, prices=prices, volumes=volumes, decision=decision,
                                  novelty=state.feature_novelty, regime=decision.regime)
        temporal = memory.attend(event)
        action = policy.decide(decision, fast_memory=state.fast_memory, slow_memory=state.slow_memory,
                               uncertainty=state.uncertainty, novelty=state.feature_novelty, regime=decision.regime,
                               temporal_up=temporal.up, temporal_down=temporal.down,
                               temporal_flat=temporal.flat, temporal_attention=temporal.attention)
        target = labeler.target(data, index)
        label_index = labeler.config.horizons.index(action.horizon)
        label = target.labels[label_index]
        labels[label] += 1
        realized = data.closes[index + action.horizon] / data.closes[index] - 1.0
        turnover = abs(action.position - previous_position)
        trade_cost = turnover * labeler.config.round_trip_cost
        net_return = action.position * realized - trade_cost
        returns.append(net_return)
        costs.append(trade_cost)
        positions.append(action.position)
        horizons.append(action.horizon)
        previous_position = action.position
        matches += temporal.matches
        attention_total += temporal.attention
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
        if learn:
            # This reward is applied only after the selected future horizon is known.
            memory.add(event, label, max(-1.0, min(1.0, net_return * 100.0)))
    economic = calculate_economic_survival_metrics(returns, positions, costs=costs)
    return CryptoEventMetrics(
        rounds=end - start, signals=signals, positive_signals=positive, negative_signals=negative,
        flat_signals=flat, total_return=economic.net_return, max_drawdown=economic.max_drawdown,
        volatility=economic.volatility,
        sharpe_like=economic.sharpe_net, hit_rate=hits / signals if signals else 0.0,
        average_position=economic.average_exposure,
        average_horizon=sum(horizons) / len(horizons) if horizons else 0.0,
        labels_up=labels[Prediction.UP], labels_down=labels[Prediction.DOWN], labels_wait=labels[Prediction.WAIT],
        economic=economic, temporal_matches=matches,
        temporal_attention=attention_total / max(1, end - start),
    )
