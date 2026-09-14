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


def run_visual_benchmark(data: BNBPredictionDataset, circuit: VisualCircuit, context: int = 32) -> tuple[VisualMetrics, VisualMetrics, VisualMetrics]:
    usable = data.size - 1
    if usable < context + 10:
        raise ValueError("dataset is too small for the visual benchmark")
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    agent = FlyVisualPredictionAgent(circuit, retina_width=context)
    train = _split(agent, data, context - 1, train_end)
    agent.reset()
    validation = _split(agent, data, train_end, validation_end)
    agent.reset()
    test = _split(agent, data, validation_end, usable)
    return train, validation, test


def _split(agent: FlyVisualPredictionAgent, data: BNBPredictionDataset, start: int, end: int) -> VisualMetrics:
    entered = correct = up = down = wait = 0
    for index in range(start, end):
        _stimulus, decision = agent.perceive(data.closes[max(0, index - 31): index + 1])
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
    return VisualMetrics(rounds, entered, correct, correct / entered if entered else 0.0, entered / rounds if rounds else 0.0, up, down, wait)
