from flydeck.causal_features import CausalFeatureBank
from flydeck.regime_detector import CausalRegimeDetector, MarketRegime
from flydeck.market_retina import BNBMarketRetina, RetinaStimulus
from flydeck.temporal_memory import DualTimescaleMemory


def test_feature_bank_is_causal_and_bounded() -> None:
    bank = CausalFeatureBank()
    first = bank.transform((100.0, 100.1, 100.2, 100.3), (10.0, 10.0, 10.0, 10.0))
    second = bank.transform((100.0, 99.8, 99.6, 99.4), (10.0, 12.0, 10.0, 14.0))
    assert first.values
    assert all(-4.0 <= value <= 4.0 for value in second.values)
    assert -1.0 <= second.volume_relative <= 1.0


def test_dual_memory_has_fast_and_slow_timescales() -> None:
    memory = DualTimescaleMemory()
    state = memory.update(1.0)
    assert state.fast > state.slow > 0.0
    memory.update(-1.0)
    assert memory.state.fast < memory.state.slow


def test_probabilistic_regime_state_sums_to_one_and_has_hysteresis() -> None:
    detector = CausalRegimeDetector()
    field = ((0.0,),)
    stimulus = RetinaStimulus(field, field, (0.0, 0.0, 0.0, 0.0), 0.9, 0.45, 0.0, 1.0, 0.2, 0.45, 0.9)
    states = [detector.step(stimulus) for _ in range(20)]
    assert abs(sum(states[-1].probabilities) - 1.0) < 1e-6
    assert states[-1].regime == MarketRegime.TREND_UP
    assert states[-1].duration >= 1
