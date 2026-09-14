from __future__ import annotations

from flydeck.finance import SyntheticCryptoMarket
from flydeck.malecns import MaleCNSEdge, MaleCNSCircuit, MaleCNSNeuron
from flydeck.malecns_v11 import build_random_topology, run_v11_diagnostics


def _circuit() -> MaleCNSCircuit:
    neurons = tuple(MaleCNSNeuron(body_id=index) for index in range(60))
    edges = tuple(MaleCNSEdge(index, (index + 1) % 60, 0.15) for index in range(60))
    inputs = tuple(tuple(range(feature * 3, feature * 3 + 3)) for feature in range(12))
    outputs = ((36, 37, 38, 39), (40, 41, 42, 43), (44, 45, 46, 47))
    return MaleCNSCircuit(neurons, edges, inputs, outputs)


def _candles():
    return SyntheticCryptoMarket(length=128, seed=123).generate()


def test_random_topology_preserves_shape_and_is_deterministic():
    first = build_random_topology(_circuit(), seed=9)
    second = build_random_topology(_circuit(), seed=9)
    assert len(first.neurons) == len(second.neurons) == 60
    assert len(first.edges) == len(second.edges) == 60
    assert first.edges == second.edges
    assert first.input_neurons == second.input_neurons
    assert first.output_neurons == second.output_neurons


def test_v11_diagnostics_runs():
    result = run_v11_diagnostics(_circuit(), _candles(), seed=7, checkpoint_interval=20)
    assert result.checkpoints
    assert result.reservoir.unique_states > 0
    assert len(result.readout.weights) == 3
    assert len(result.readout.biases) == 3
    assert sum(result.trained_test_actions) > 0
    assert sum(result.random_topology_test_actions) > 0
    assert result.random_topology_edges == 60


def test_v11_diagnostics_is_deterministic():
    first = run_v11_diagnostics(_circuit(), _candles(), seed=7, checkpoint_interval=20)
    second = run_v11_diagnostics(_circuit(), _candles(), seed=7, checkpoint_interval=20)
    assert first == second
