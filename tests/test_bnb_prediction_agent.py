from __future__ import annotations

from flydeck.bnb_prediction import Prediction
from flydeck.bnb_prediction_agent import BNBFeatureSensor, BNBObservation, FlyBNBPredictionAgent, FlyDecision
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset, run_bnb_prediction_benchmark
from flydeck.malecns import MaleCNSCircuit, MaleCNSEdge, MaleCNSNeuron


def make_circuit() -> MaleCNSCircuit:
    neurons = tuple(MaleCNSNeuron(i) for i in range(36))
    edges = tuple(MaleCNSEdge(i, (i + 1) % 36, 0.01) for i in range(36))
    inputs = tuple((i,) for i in range(12))
    outputs = ((12,), (13,), (14,))
    return MaleCNSCircuit(neurons, edges, inputs, outputs)


def test_sensor_is_causal_and_fixed_size() -> None:
    sensor = BNBFeatureSensor()
    observation = BNBObservation(1, tuple(100 + i for i in range(25)), tuple(10 for _ in range(25)))
    features = sensor.encode(observation)
    assert len(features) == 12
    assert all(-1.0 <= value <= 1.0 for value in features)


def test_agent_has_only_up_down_wait_behaviors() -> None:
    agent = FlyBNBPredictionAgent(make_circuit(), seed=1, confidence_threshold=0.0)
    observation = BNBObservation(1, tuple(100.0 + i * 0.1 for i in range(25)), tuple(10.0 for _ in range(25)))
    prediction = agent.predict(observation)
    assert prediction.decision in (FlyDecision.UP, FlyDecision.DOWN, FlyDecision.WAIT)


def test_dataset_outcome_uses_next_candle_only() -> None:
    data = BNBPredictionDataset(
        timestamps=tuple(range(4)), opens=(1, 2, 3, 4), highs=(1, 2, 3, 4),
        lows=(1, 2, 3, 4), closes=(10, 11, 9, 9), volumes=(1, 1, 1, 1)
    )
    assert data.outcome(0) == Prediction.UP
    assert data.outcome(1) == Prediction.DOWN
    assert data.outcome(2) == Prediction.WAIT
