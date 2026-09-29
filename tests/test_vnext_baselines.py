from flydeck.baseline_models import AdaBoostStumpBaseline, LogisticRegressionBaseline, causal_feature_vector
from flydeck.benchmark_vnext import run_vnext_baselines
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset


def _data(n: int = 500) -> BNBPredictionDataset:
    closes = []
    price = 100.0
    for i in range(n):
        direction = 1.0 if (i // 8) % 2 == 0 else -1.0
        price += direction * (0.15 + (i % 3) * 0.02)
        closes.append(price)
    close_tuple = tuple(closes)
    return BNBPredictionDataset(
        timestamps=tuple(i * 300_000 for i in range(n)),
        opens=tuple(v - 0.02 for v in close_tuple),
        highs=tuple(v + 0.10 for v in close_tuple),
        lows=tuple(v - 0.10 for v in close_tuple),
        closes=close_tuple,
        volumes=tuple(100.0 + (i % 7) for i in range(n)),
    )


def test_features_are_finite_and_causal_shape_is_stable() -> None:
    row = causal_feature_vector(_data(), 40)
    assert len(row) == 24
    assert all(value == value for value in row)


def test_dependency_free_ml_baselines_fit_and_predict() -> None:
    data = _data()
    for model in (LogisticRegressionBaseline(epochs=20), AdaBoostStumpBaseline(estimators=4)):
        model.fit(data, 31, 300)
        probability = model.predict_probability(data, 320)
        assert 0.0 <= probability <= 1.0


def test_vnext_benchmark_freezes_validation_thresholds() -> None:
    result = run_vnext_baselines(
        _data(),
        context=32,
        purge=1,
        min_validation_entries=20,
        min_coverage=0.20,
    )
    assert result.frozen_thresholds
    assert any(row.name.endswith("+ WAIT") for row in result.test)\n    assert all(len(item) == 3 for item in result.frozen_thresholds)
    for row in result.test:
        assert 0.0 <= row.coverage <= 1.0
        assert 0.0 <= row.accuracy <= 1.0
