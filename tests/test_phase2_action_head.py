from flydeck.action_head import ActionProbabilityHead
from flydeck.bnb_prediction import Prediction
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.bnb_visual_runner import run_visual_benchmark


def test_action_head_probabilities_sum_to_one_and_include_wait() -> None:
    head = ActionProbabilityHead(initial_temperature=2.0)
    probabilities = head.predict(0.8, 0.1, 0.05)
    assert abs(sum(probabilities.as_tuple()) - 1.0) < 1e-9
    assert probabilities.up > probabilities.down
    assert 0.0 < probabilities.wait < 1.0


def test_action_head_updates_temperature_only_with_causal_outcome() -> None:
    head = ActionProbabilityHead(initial_temperature=2.0)
    before = head.temperature
    head.predict(0.8, 0.1, 0.05)
    head.observe(Prediction.UP)
    assert head.temperature != before
    frozen = head.temperature
    head.set_learning(False)
    head.predict(0.1, 0.8, 0.05)
    head.observe(Prediction.DOWN)
    assert head.temperature == frozen


def test_benchmark_exposes_multiclass_calibration() -> None:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    circuit = VisualCircuit(neurons=neurons, edges=(VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)),
                            l1_inputs=(0,), l2_inputs=(1,), t4_outputs=((), (), (2,), (3,)),
                            t5_outputs=((), (), (4,), (5,)), spatial_mode="soma_xy_proxy")
    closes = tuple(100.0 + i * 0.1 for i in range(80))
    data = BNBPredictionDataset(tuple(range(80)), closes, closes, closes, closes, (10.0,) * 80)
    result = run_visual_benchmark(data, circuit, context=8)
    assert all(metrics.multiclass_brier_score >= 0.0 for metrics in result)
    assert all(metrics.multiclass_expected_calibration_error >= 0.0 for metrics in result)
