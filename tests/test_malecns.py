from __future__ import annotations

from flydeck.finance_malecns import train_malecns_synthetic
from flydeck.malecns import MaleCNSEdge, MaleCNSCircuit, MaleCNSNeuron, MaleCNSReservoir


def _circuit() -> MaleCNSCircuit:
    neurons = tuple(MaleCNSNeuron(body_id=index) for index in range(60))
    edges = tuple(
        MaleCNSEdge(index, (index + 1) % 60, 0.15)
        for index in range(60)
    )
    inputs = tuple(tuple(range(feature * 3, feature * 3 + 3)) for feature in range(12))
    outputs = (
        (36, 37, 38, 39),
        (40, 41, 42, 43),
        (44, 45, 46, 47),
    )
    return MaleCNSCircuit(neurons, edges, inputs, outputs)


def test_malecns_reservoir_is_deterministic():
    circuit = _circuit()
    first = MaleCNSReservoir(circuit, feature_count=12, seed=7)
    second = MaleCNSReservoir(circuit, feature_count=12, seed=7)
    observation = tuple(0.1 for _ in range(12))
    assert first.step(observation) == second.step(observation)


def test_malecns_state_changes_and_readout_updates():
    reservoir = MaleCNSReservoir(_circuit(), feature_count=12, seed=7)
    scores_before = reservoir.step(tuple(0.5 for _ in range(12)))
    activity = reservoir.action_activity(0)
    reservoir.update_readout(0, 0.5, learning_rate=0.01, activity=activity)
    scores_after = reservoir.action_scores()
    assert scores_before != scores_after
    assert reservoir.neuron_count == 60
    assert reservoir.edge_count == 60


def test_malecns_crypto_training_runs():
    result = train_malecns_synthetic(
        _circuit(),
        episodes=2,
        market_length=64,
        max_steps=20,
        seed=7,
    )
    assert result.episodes == 2
    assert result.total_actions == 40
    assert result.total_trades >= 0
