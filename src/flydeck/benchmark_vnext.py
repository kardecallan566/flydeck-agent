from __future__ import annotations

from dataclasses import dataclass

from .baseline_models import AdaBoostStumpBaseline, LogisticRegressionBaseline
from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .probability_calibration import fit_platt_calibrator
from .scientific_benchmarks import (
    AlwaysDownBaseline,
    AlwaysUpBaseline,
    MomentumBaseline,
    PreviousDirectionBaseline,
)
from .selective_policy import calibrate_selective_policy
from .statistical_metrics import BinaryEvaluation, evaluate_binary
from .validation_protocol import ChronologicalProtocol, IndexRange, build_chronological_protocol


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    protocol: ChronologicalProtocol
    validation: tuple[BinaryEvaluation, ...]
    test: tuple[BinaryEvaluation, ...]
    frozen_thresholds: tuple[tuple[str, float, float], ...]


def _outcomes(dataset: BNBPredictionDataset, span: IndexRange) -> tuple[Prediction, ...]:
    return tuple(dataset.outcome(index) for index in range(span.start, span.end))


def _hard_predictions(model: object, dataset: BNBPredictionDataset, span: IndexRange) -> tuple[Prediction, ...]:
    return tuple(model.predict(dataset, index) for index in range(span.start, span.end))


def _probabilities(model: object, dataset: BNBPredictionDataset, span: IndexRange) -> tuple[float, ...]:
    return tuple(float(model.predict_probability(dataset, index)) for index in range(span.start, span.end))


def _hard_from_probabilities(probabilities: tuple[float, ...]) -> tuple[Prediction, ...]:
    return tuple(Prediction.UP if value >= 0.5 else Prediction.DOWN for value in probabilities)


def run_vnext_baselines(
    dataset: BNBPredictionDataset,
    *,
    context: int = 32,
    purge: int = 1,
    min_validation_entries: int = 100,
    min_coverage: float = 0.10,
) -> BenchmarkResult:
    """Leakage-resistant baselines with held-out calibration and asymmetric WAIT.

    The validation interval is split chronologically in two. The first half
    calibrates probabilities; the second half selects UP/DOWN abstention
    thresholds. Both are then frozen before the OOS test.
    """
    protocol = build_chronological_protocol(dataset.size, context=context, purge=purge)
    validation_rows: list[BinaryEvaluation] = []
    test_rows: list[BinaryEvaluation] = []
    thresholds: list[tuple[str, float, float]] = []

    deterministic = (
        ("Always UP", AlwaysUpBaseline()),
        ("Always DOWN", AlwaysDownBaseline()),
        ("Lag-1 Persistence", PreviousDirectionBaseline()),
        ("Momentum-6", MomentumBaseline(window=6)),
    )
    validation_outcomes = _outcomes(dataset, protocol.validation)
    test_outcomes = _outcomes(dataset, protocol.test)
    for name, model in deterministic:
        val_preds = _hard_predictions(model, dataset, protocol.validation)
        test_preds = _hard_predictions(model, dataset, protocol.test)
        validation_rows.append(evaluate_binary(name, val_preds, validation_outcomes))
        test_rows.append(evaluate_binary(name, test_preds, test_outcomes))

    probabilistic = (
        ("Logistic OHLCV+", LogisticRegressionBaseline()),
        ("AdaBoost Stumps OHLCV+", AdaBoostStumpBaseline()),
    )
    for name, model in probabilistic:
        model.fit(dataset, protocol.train.start, protocol.train.end)
        val_raw = _probabilities(model, dataset, protocol.validation)
        test_raw = _probabilities(model, dataset, protocol.test)

        validation_rows.append(
            evaluate_binary(name, _hard_from_probabilities(val_raw), validation_outcomes, p_up=val_raw)
        )
        test_rows.append(
            evaluate_binary(name, _hard_from_probabilities(test_raw), test_outcomes, p_up=test_raw)
        )

        split = max(1, len(val_raw) // 2)
        calibrator = fit_platt_calibrator(val_raw[:split], validation_outcomes[:split])
        val_prob = calibrator.apply(val_raw)
        test_prob = calibrator.apply(test_raw)

        validation_rows.append(
            evaluate_binary(name + " + CAL", _hard_from_probabilities(val_prob), validation_outcomes, p_up=val_prob)
        )
        test_rows.append(
            evaluate_binary(name + " + CAL", _hard_from_probabilities(test_prob), test_outcomes, p_up=test_prob)
        )

        calibrated = calibrate_selective_policy(
            val_prob[split:],
            validation_outcomes[split:],
            min_entries=min_validation_entries,
            min_coverage=min_coverage,
            asymmetric=True,
        )
        thresholds.append((name, calibrated.policy.up_threshold, calibrated.policy.down_threshold))
        validation_rows.append(
            evaluate_binary(
                name + " + CAL + WAIT",
                calibrated.policy.apply(val_prob),
                validation_outcomes,
                p_up=val_prob,
            )
        )
        test_rows.append(
            evaluate_binary(
                name + " + CAL + WAIT",
                calibrated.policy.apply(test_prob),
                test_outcomes,
                p_up=test_prob,
            )
        )

    return BenchmarkResult(
        protocol=protocol,
        validation=tuple(validation_rows),
        test=tuple(test_rows),
        frozen_thresholds=tuple(thresholds),
    )
