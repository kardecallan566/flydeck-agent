from flydeck.bnb_prediction import Prediction
from flydeck.statistical_metrics import evaluate_binary, wilson_interval


def test_wilson_interval_requires_sample_support() -> None:
    small = wilson_interval(6, 10)
    large = wilson_interval(600, 1000)
    assert small[0] < 0.60 < small[1]
    assert large[0] < 0.60 < large[1]
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_binary_metrics_include_balanced_accuracy_and_brier() -> None:
    outcomes = (Prediction.UP, Prediction.UP, Prediction.DOWN, Prediction.DOWN)
    predictions = (Prediction.UP, Prediction.DOWN, Prediction.DOWN, Prediction.WAIT)
    probabilities = (0.8, 0.4, 0.2, 0.45)
    metrics = evaluate_binary("x", predictions, outcomes, p_up=probabilities)
    assert metrics.entered == 3
    assert metrics.correct == 2
    assert 0.0 <= metrics.balanced_accuracy <= 1.0
    assert metrics.brier_score is not None
