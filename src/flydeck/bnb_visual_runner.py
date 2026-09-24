from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class VisualMetrics:
    rounds: int
    entered: int
    correct: int
    accuracy: float
    coverage: float
    up: int
    down: int
    wait: int
    brier_score: float
    expected_calibration_error: float
    regimes: tuple[tuple[str, int], ...]


def run_visual_benchmark(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    context: int = 32,
) -> tuple[VisualMetrics, VisualMetrics, VisualMetrics]:
    if context < 4:
        raise ValueError("context must be at least four candles")
    usable = data.size - 1
    if usable < context + 10:
        raise ValueError("dataset is too small for the visual benchmark")
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    agent = FlyVisualPredictionAgent(circuit, retina_width=context)
    train = _split(agent, data, context - 1, train_end, context)
    agent.set_learning(False)
    agent.reset(preserve_learning=True)
    validation = _split(agent, data, train_end, validation_end, context)
    agent.reset(preserve_learning=True)
    test = _split(agent, data, validation_end, usable, context)
    return train, validation, test


def _split(
    agent: FlyVisualPredictionAgent,
    data: BNBPredictionDataset,
    start: int,
    end: int,
    context: int,
) -> VisualMetrics:
    entered = correct = up = down = wait = 0
    brier_total = 0.0
    calibration: list[tuple[float, int]] = []
    regime_counts: dict[str, int] = {}
    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1) : index + 1]
        volumes = data.volumes[max(0, index - context + 1) : index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=volumes)
        outcome = data.outcome(index)
        regime_counts[decision.regime] = regime_counts.get(decision.regime, 0) + 1
        if decision.wait:
            wait += 1
        elif decision.up_score > decision.down_score:
            up += 1
            entered += 1
            is_correct = int(outcome == Prediction.UP)
            correct += is_correct
            p_direction = decision.p_up / max(1e-12, decision.p_up + decision.p_down)
            brier_total += (p_direction - float(outcome == Prediction.UP)) ** 2
            calibration.append((max(decision.p_up, decision.p_down) / max(1e-12, decision.p_up + decision.p_down), is_correct))
        else:
            down += 1
            entered += 1
            is_correct = int(outcome == Prediction.DOWN)
            correct += is_correct
            p_direction = decision.p_down / max(1e-12, decision.p_up + decision.p_down)
            brier_total += (p_direction - float(outcome == Prediction.DOWN)) ** 2
            calibration.append((max(decision.p_up, decision.p_down) / max(1e-12, decision.p_up + decision.p_down), is_correct))
    rounds = end - start
    brier = brier_total / entered if entered else 0.0
    ece = _expected_calibration_error(calibration)
    return VisualMetrics(
        rounds=rounds,
        entered=entered,
        correct=correct,
        accuracy=correct / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        up=up,
        down=down,
        wait=wait,
        brier_score=brier,
        expected_calibration_error=ece,
        regimes=tuple(sorted(regime_counts.items())),
    )


def _expected_calibration_error(values: list[tuple[float, int]], bins: int = 10) -> float:
    if not values:
        return 0.0
    total = len(values)
    error = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        members = [item for item in values if lower <= item[0] < upper or (bucket == bins - 1 and item[0] <= upper)]
        if members:
            confidence = sum(item[0] for item in members) / len(members)
            accuracy = sum(item[1] for item in members) / len(members)
            error += len(members) / total * abs(confidence - accuracy)
    return error
