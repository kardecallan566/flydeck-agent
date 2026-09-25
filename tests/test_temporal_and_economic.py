from flydeck.bnb_prediction import Prediction
from flydeck.crypto_event_policy import CryptoEventConfig
from flydeck.economic_metrics import calculate_economic_survival_metrics
from flydeck.temporal_events import TemporalEventExtractor
from flydeck.temporal_memory import SparseTemporalMemory


def test_sparse_temporal_memory_retrieves_only_causal_events() -> None:
    from flydeck.visual_agent import VisualDecision
    decision = VisualDecision(up_score=0.7, down_score=0.1, confidence=0.7, wait=False,
                              consensus=0.7, conflict=0.0, arousal=0.2, fast_bias=0.2,
                              slow_bias=0.1, action=Prediction.UP, reason="COMMIT_UP",
                              p_wait=0.1, p_up=0.7, p_down=0.2, regime="TREND_UP")
    extractor = TemporalEventExtractor()
    event = extractor.extract(timestamp=10, prices=(100.0, 100.2, 100.4), volumes=(1.0, 1.0, 2.0),
                               decision=decision, regime="TREND_UP")
    memory = SparseTemporalMemory(capacity=8, top_k=2)
    memory.add(event, Prediction.UP, 0.5)
    context = memory.attend(event)
    assert context.matches == 1
    assert context.up > 0.0


def test_economic_metrics_penalize_costs_and_report_drawdown() -> None:
    result = calculate_economic_survival_metrics([0.01, -0.02, 0.005], [0.2, -0.2, 0.1], costs=[0.001, 0.001, 0.001])
    assert result.net_return < result.gross_return
    assert result.max_drawdown > 0.0
    assert result.cvar_95 < 0.0
    assert result.turnover > 0.0
