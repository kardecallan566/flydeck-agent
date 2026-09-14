from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction
from .bnb_prediction_agent import FlyBNBPredictionAgent, FlyDecision
from .bnb_prediction_data import BNBPredictionDataset


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    rounds: int
    entered: int
    skipped: int
    correct: int
    accuracy: float
    coverage: float
    up: int
    down: int
    wait: int


@dataclass(frozen=True, slots=True)
class BNBPredictionBenchmark:
    train: PredictionMetrics
    validation: PredictionMetrics
    test: PredictionMetrics


def _evaluate(agent: FlyBNBPredictionAgent, data: BNBPredictionDataset, start: int, end: int, learn: bool) -> PredictionMetrics:
    rounds = max(0, end - start)
    entered = correct = up = down = wait = 0
    for index in range(start, end):
        prediction = agent.predict(data.observation(index))
        if prediction.decision == FlyDecision.UP:
            up += 1
            entered += 1
        elif prediction.decision == FlyDecision.DOWN:
            down += 1
            entered += 1
        else:
            wait += 1
        outcome = data.round(index).outcome
        if prediction.decision != FlyDecision.WAIT:
            if prediction.decision == FlyDecision.UP and outcome == Prediction.UP:
                correct += 1
            elif prediction.decision == FlyDecision.DOWN and outcome == Prediction.DOWN:
                correct += 1
        if learn:
            agent.learn(outcome)
    return PredictionMetrics(
        rounds=rounds,
        entered=entered,
        skipped=wait,
        correct=correct,
        accuracy=correct / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        up=up,
        down=down,
        wait=wait,
    )


def run_bnb_prediction_benchmark(
    data: BNBPredictionDataset,
    seed: int = 123,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    confidence_threshold: float = 0.20,
) -> BNBPredictionBenchmark:
    usable = data.size - 1
    if usable < 10:
        raise ValueError("BNB dataset needs at least 11 candles")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("invalid chronological split ratios")

    train_end = int(usable * train_ratio)
    validation_end = train_end + int(usable * validation_ratio)
    agent = FlyBNBPredictionAgent(
        circuit=_require_circuit(data),
        seed=seed,
        confidence_threshold=confidence_threshold,
    )
    train = _evaluate(agent, data, 0, train_end, learn=True)
    validation = _evaluate(agent, data, train_end, validation_end, learn=False)
    test = _evaluate(agent, data, validation_end, usable, learn=False)
    return BNBPredictionBenchmark(train, validation, test)


def _require_circuit(data: BNBPredictionDataset):
    circuit = getattr(data, "circuit", None)
    if circuit is None:
        raise ValueError("dataset must be bound to a MaleCNS circuit")
    return circuit
