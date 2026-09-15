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
    agent.reset()
    validation = _split(agent, data, train_end, validation_end, context)
    agent.reset()
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
    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1) : index + 1]
        _stimulus, decision = agent.perceive(prices)
        outcome = data.outcome(index)
        if decision.wait:
            wait += 1
        elif decision.up_score > decision.down_score:
            up += 1
            entered += 1
            correct += int(outcome == Prediction.UP)
        else:
            down += 1
            entered += 1
            correct += int(outcome == Prediction.DOWN)
    rounds = end - start
    return VisualMetrics(
        rounds=rounds,
        entered=entered,
        correct=correct,
        accuracy=correct / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        up=up,
        down=down,
        wait=wait,
    )
