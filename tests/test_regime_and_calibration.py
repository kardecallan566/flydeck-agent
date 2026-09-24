from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.market_retina import BNBMarketRetina
from flydeck.regime_detector import CausalRegimeDetector, MarketRegime
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron
from flydeck.bnb_visual_runner import run_visual_benchmark


def _circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=(VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)),
        l1_inputs=(0,), l2_inputs=(1,),
        t4_outputs=((), (), (2,), (3,)), t5_outputs=((), (), (4,), (5,)),
        spatial_mode="soma_xy_proxy",
    )


def _data(n: int = 80) -> BNBPredictionDataset:
    closes = tuple(100.0 + i * 0.08 + (0.2 if i % 7 == 0 else 0.0) for i in range(n))
    return BNBPredictionDataset(
        timestamps=tuple(range(n)), opens=closes,
        highs=tuple(v + 0.1 for v in closes), lows=tuple(v - 0.1 for v in closes),
        closes=closes, volumes=(10.0,) * n,
    )


def test_regime_detector_is_causal_and_returns_known_regimes() -> None:
    retina = BNBMarketRetina(width=8, height=6)
    detector = CausalRegimeDetector()
    states = [detector.step(retina.encode(tuple(100.0 + i for i in range(8)))) for _ in range(3)]
    assert all(state.regime in tuple(MarketRegime) for state in states)
    assert detector.state == states[-1]
    detector.reset()
    assert detector.state.regime == MarketRegime.RANGE


def test_visual_metrics_include_brier_ece_and_regimes() -> None:
    train, validation, test = run_visual_benchmark(_data(), _circuit(), context=8)
    for metrics in (train, validation, test):
        assert metrics.brier_score >= 0.0
        assert metrics.expected_calibration_error >= 0.0
        assert sum(count for _regime, count in metrics.regimes) == metrics.rounds


def test_detector_identifies_persistent_up_and_down_motion() -> None:
    from flydeck.market_retina import RetinaStimulus

    def stimulus(velocity: float, acceleration: float = 0.0) -> RetinaStimulus:
        field = ((0.0,),)
        return RetinaStimulus(field, field, (0.0, 0.0, 0.0, 0.0), 0.9, velocity, acceleration, 1.0, 0.2, velocity, 0.9)

    detector = CausalRegimeDetector()
    up_states = [detector.step(stimulus(0.45)) for _ in range(20)]
    assert up_states[-1].regime == MarketRegime.TREND_UP
    detector.reset()
    down_states = [detector.step(stimulus(-0.45)) for _ in range(20)]
    assert down_states[-1].regime == MarketRegime.TREND_DOWN
