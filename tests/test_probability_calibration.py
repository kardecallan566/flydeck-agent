from flydeck.bnb_prediction import Prediction
from flydeck.probability_calibration import PlattCalibrator, fit_platt_calibrator


def test_platt_calibrator_preserves_probability_bounds() -> None:
    calibrator = PlattCalibrator(slope=1.2, intercept=-0.1)
    values = calibrator.apply((0.0, 0.2, 0.5, 0.8, 1.0))
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values == tuple(sorted(values))


def test_platt_fit_uses_binary_outcomes_and_returns_finite_mapping() -> None:
    probabilities = tuple([0.70, 0.30, 0.60, 0.40] * 20)
    outcomes = tuple([Prediction.UP, Prediction.DOWN, Prediction.UP, Prediction.DOWN] * 20)
    calibrator = fit_platt_calibrator(probabilities, outcomes, epochs=50)
    mapped = calibrator.apply(probabilities)
    assert all(value == value for value in mapped)
    assert sum(mapped[::4]) / len(mapped[::4]) > 0.5
    assert sum(mapped[1::4]) / len(mapped[1::4]) < 0.5
