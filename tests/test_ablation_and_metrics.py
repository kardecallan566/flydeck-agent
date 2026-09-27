from flydeck.bnb_prediction import Prediction
from flydeck.standard_metrics import standardize_metrics
from flydeck.temporal_events import TemporalEvent
from flydeck.temporal_memory import SparseTemporalMemory


def _event(timestamp: int) -> TemporalEvent:
    return TemporalEvent(timestamp=timestamp, state=(0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.0),
                         direction=0.6, intensity=0.8, volatility=0.001,
                         volume_surprise=0.1, novelty=0.0, regime="RANGE")


def test_frozen_train_memory_has_matches_in_later_test_split() -> None:
    memory = SparseTemporalMemory(decay=0.985, long_term_floor=0.10, novelty_threshold=0.03)
    memory.add(_event(100), Prediction.UP, 0.5)
    context = memory.attend(_event(50_000))
    assert context.matches > 0


def test_standard_metrics_have_one_definition() -> None:
    metrics = standardize_metrics(rounds=10, entered=4, correct=3, profitable=2,
                                  net_returns=[0.01, -0.02])
    assert metrics.accuracy == 0.75
    assert metrics.hit_rate == 0.5
    assert metrics.coverage == 0.4
    assert metrics.economic_return < 0.0
