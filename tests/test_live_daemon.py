"""Unit tests for FlyDeckLiveDaemon with mock provider."""
from pathlib import Path
import pytest

from flydeck.data import MarketCandle, MarketDataProvider, MarketDataset, MarketDataService, dataset_from_candles
from flydeck.live_daemon import FlyDeckLiveDaemon
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit, VisualNeuron, VisualEdge


def _toy_circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.2, 0.2),
        VisualNeuron(2, "L2", "visual_entry", "gaba", -1.0, 0.8, 0.8),
        VisualNeuron(3, "T4", "motion_t4", "acetylcholine", 1.0, 0.2, 0.5),
        VisualNeuron(4, "T4", "motion_t4", "acetylcholine", 1.0, 0.2, 0.5),
        VisualNeuron(5, "T5", "motion_t5", "acetylcholine", 1.0, 0.8, 0.5),
        VisualNeuron(6, "T5", "motion_t5", "acetylcholine", 1.0, 0.8, 0.5),
    )
    edges = (
        VisualEdge(0, 2, 1.0),
        VisualEdge(1, 4, 1.0),
        VisualEdge(2, 3, 0.5),
        VisualEdge(4, 5, 0.5),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=edges,
        l1_inputs=(0,),
        l2_inputs=(1,),
        t4_outputs=((), (), (2,), (3,)),
        t5_outputs=((), (), (4,), (5,)),
        spatial_mode="soma_xy_proxy",
    )


class MockProvider(MarketDataProvider):
    name = "mock"

    def __init__(self, candle_count: int = 40) -> None:
        self.candles = [
            MarketCandle(
                timestamp=i * 300_000,
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=15.0,
            )
            for i in range(candle_count)
        ]

    def fetch(self, symbol: str, interval: str, limit: int = 1000) -> tuple[MarketCandle, ...]:
        return tuple(self.candles[-limit:])


def test_daemon_time_to_next_boundary() -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)
    daemon = FlyDeckLiveDaemon(agent, interval_seconds=300)

    # If current time is 100s into a 300s window (e.g. 100), next boundary is 300
    wait_time = daemon.time_to_next_boundary(current_time=100.0, buffer_seconds=2.0)
    assert abs(wait_time - (200.0 + 2.0)) < 1e-3


def test_daemon_lead_time_calculation() -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)
    daemon = FlyDeckLiveDaemon(agent, interval_seconds=300, lead_time_seconds=25.0)

    # If current time is 100s into 300s window, next trigger is at 300 - 25 = 275s -> wait 175s
    wait_time = daemon.time_to_next_boundary(current_time=100.0)
    assert abs(wait_time - 175.0) < 1e-3

    # If current time is 280s (already past 275s for the 300s boundary), target is 600 - 25 = 575s -> wait 295s
    wait_time_late = daemon.time_to_next_boundary(current_time=280.0)
    assert abs(wait_time_late - 295.0) < 1e-3


def test_daemon_step_execution_and_checkpointing(tmp_path: Path) -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)

    mock_prov = MockProvider(candle_count=40)
    service = MarketDataService(mock_prov, cache_root=tmp_path / "cache")
    ckpt_file = tmp_path / "daemon_ckpt.json"
    diag_file = tmp_path / "daemon_diag.jsonl"

    daemon = FlyDeckLiveDaemon(
        agent=agent,
        context_window=10,
        checkpoint_file=ckpt_file,
        diagnostics_file=diag_file,
        service=service,
    )

    record = daemon.execute_step()
    assert record is not None
    assert record["round"] == 1
    assert "action" in record
    assert "close" in record
    assert ckpt_file.exists()
    assert diag_file.exists()

    # Second call with same latest timestamp should return None (deduplication)
    duplicate = daemon.execute_step()
    assert duplicate is None


def test_daemon_triggers_on_action_callback(tmp_path: Path) -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)
    mock_prov = MockProvider(candle_count=40)
    service = MarketDataService(mock_prov, cache_root=tmp_path / "cache")

    received_actions = []

    def handle_action(action: str, record: dict) -> None:
        received_actions.append((action, record["close"]))

    daemon = FlyDeckLiveDaemon(
        agent=agent,
        context_window=10,
        checkpoint_file=tmp_path / "ckpt.json",
        diagnostics_file=tmp_path / "diag.jsonl",
        service=service,
        on_action=handle_action,
    )

    record = daemon.execute_step()
    assert len(received_actions) == 1
    assert received_actions[0][0] in ("UP", "DOWN", "WAIT")
