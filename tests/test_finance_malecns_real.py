from __future__ import annotations

from flydeck.finance import SyntheticCryptoMarket
from flydeck.finance_malecns_real import RealMarketDataset, run_malecns_real_benchmark
from flydeck.malecns import MaleCNSEdge, MaleCNSCircuit, MaleCNSNeuron


def _circuit() -> MaleCNSCircuit:
    neurons = tuple(MaleCNSNeuron(body_id=index) for index in range(60))
    edges = tuple(MaleCNSEdge(index, (index + 1) % 60, 0.15) for index in range(60))
    inputs = tuple(tuple(range(feature * 3, feature * 3 + 3)) for feature in range(12))
    outputs = ((36, 37, 38, 39), (40, 41, 42, 43), (44, 45, 46, 47))
    return MaleCNSCircuit(neurons, edges, inputs, outputs)


def _dataset(length: int = 128) -> RealMarketDataset:
    candles = SyntheticCryptoMarket(length=length, seed=123).generate()
    timestamps = tuple(1_700_000_000_000 + i * 3_600_000 for i in range(length))
    return RealMarketDataset("BTCUSDT", "1h", candles, timestamps, "test")


def test_malecns_real_benchmark_runs():
    result = run_malecns_real_benchmark(_circuit(), _dataset(), seed=7)
    assert result.train.steps > 0
    assert result.validation.steps > 0
    assert result.test.steps > 0
    assert result.random_test.steps == result.test.steps
    assert sum(result.test.action_counts) == result.test.steps


def test_malecns_real_benchmark_is_deterministic():
    first = run_malecns_real_benchmark(_circuit(), _dataset(), seed=7)
    second = run_malecns_real_benchmark(_circuit(), _dataset(), seed=7)
    assert first == second
