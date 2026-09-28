from flydeck.bnb_prediction import Prediction
from flydeck.selective_policy import SelectivePolicy, calibrate_selective_policy


def test_selective_policy_abstains_near_half() -> None:
    policy = SelectivePolicy(confidence_threshold=0.20)
    assert policy.decide(0.51) == Prediction.WAIT
    assert policy.decide(0.80) == Prediction.UP
    assert policy.decide(0.20) == Prediction.DOWN


def test_calibration_respects_minimum_support() -> None:
    probabilities = tuple([0.90, 0.10] * 60 + [0.52, 0.48] * 40)
    outcomes = tuple([Prediction.UP, Prediction.DOWN] * 60 + [Prediction.DOWN, Prediction.UP] * 40)
    result = calibrate_selective_policy(
        probabilities,
        outcomes,
        min_entries=100,
        min_coverage=0.40,
    )
    assert result.validation.entered >= 100
    assert result.validation.coverage >= 0.40
    assert 0.0 <= result.policy.confidence_threshold <= 1.0
