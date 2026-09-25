from flydeck.bnb_prediction import Prediction
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.crypto_event_policy import CryptoEventConfig, CryptoEventLabeler, CryptoEventPolicy
from flydeck.visual_agent import VisualDecision


def _data() -> BNBPredictionDataset:
    closes = tuple(100.0 + i * 0.5 for i in range(40))
    return BNBPredictionDataset(tuple(range(40)), closes, closes, closes, closes, (10.0,) * 40)


def _decision() -> VisualDecision:
    return VisualDecision(up_score=0.8, down_score=0.1, confidence=0.8, wait=False,
                          consensus=0.8, conflict=0.0, arousal=0.2, fast_bias=0.4,
                          slow_bias=0.2, action=Prediction.UP, reason="COMMIT_UP",
                          p_wait=0.05, p_up=0.8, p_down=0.15, regime="TREND_UP")


def test_crypto_event_labels_include_cost_and_horizons() -> None:
    config = CryptoEventConfig(horizons=(1, 3, 6), fee_bps=5.0, slippage_bps=2.0)
    target = CryptoEventLabeler(config).target(_data(), 10)
    assert len(target.horizon_returns) == 3
    assert len(target.thresholds) == 3
    assert target.labels[-1] == Prediction.UP
    assert target.thresholds[0] >= config.round_trip_cost


def test_crypto_event_policy_returns_continuous_position() -> None:
    action = CryptoEventPolicy().decide(_decision(), fast_memory=0.5, slow_memory=0.2,
                                         uncertainty=0.1, novelty=0.2, regime="TREND_UP")
    assert 0.0 < action.position <= 1.0
    assert action.direction > 0.0
    assert action.horizon in (1, 3, 6)


def test_crypto_event_policy_reduces_exposure_but_does_not_veto_shock() -> None:
    decision = _decision()
    normal = CryptoEventPolicy().decide(decision, uncertainty=0.1, novelty=0.0, regime="RANGE")
    shock = CryptoEventPolicy().decide(decision, uncertainty=0.1, novelty=0.0, regime="SHOCK")
    assert shock.position > 0.0
    assert abs(shock.position) < abs(normal.position)
