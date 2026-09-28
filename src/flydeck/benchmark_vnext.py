from __future__ import annotations

from dataclasses import dataclass

from .baseline_models import AdaBoostStumpBaseline, LogisticRegressionBaseline
from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
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
    frozen_thresholds: tuple[tuple[str, float], ...]


def _outcomes(dataset: BNBPredictionDataset, span: IndexRange) -> tuple[Prediction, ...]:
    return tuple(dataset.outcome(index) for index in range(span.start, span.end))


def _hard_predictions(model: object, dataset: BNBPredictionDataset, span: IndexRange) -> tuple[Prediction, ...]:
    return tuple(model.predict(dataset, index) for index in range(span.start, span.end))


def _probabilities(model: object, dataset: BNBPredictionDataset, span: IndexRange) -> tuple[float, ...]:
    return tuple(float(model.predict_probability(dataset, index)) for index in range(span.start, span.end))


def run_vnext_baselines(
    dataset: BNBPredictionDataset,
    *,
    context: int = 32,
    purge: int = 1,
    min_validation_entries: int = 100,
    min_coverage: float = 0.10,
) -> BenchmarkResult:
    """Leakage-resistant baseline battery with validation-only WAIT calibration."""
    protocol = build_chronological_protocol(dataset.size, context=context, purge=purge)
    validation_rows: list[BinaryEvaluation] = []
    test_rows: list[BinaryEvaluation] = []
    thresholds: list[tuple[str, float]] = []

    deterministic = (
        ("Always UP", AlwaysUpBaseline()),
        ("Always DOWN", AlwaysDownBaseline()),
        ("Lag-1 Persistence", PreviousDirectionBaseline()),
        ("Momentum-6", MomentumBaseline(window=6)),
    )
    for name, model in deterministic:
        val_preds = _hard_predictions(model, dataset, protocol.validation)
        test_preds = _hard_predictions(model, dataset, protocol.test)
        validation_rows.append(evaluate_binary(name, val_preds, _outcomes(dataset, protocol.validation)))
        test_rows.append(evaluate_binary(name, test_preds, _outcomes(dataset, protocol.test)))

    probabilistic = (
        ("Logistic OHLCV", LogisticRegressionBaseline()),
        ("AdaBoost Stumps OHLCV", AdaBoostStumpBaseline()),
    )
    for name, model in probabilistic:
        model.fit(dataset, protocol.train.start, protocol.train.end)
        val_prob = _probabilities(model, dataset, protocol.validation)
        test_prob = _probabilities(model, dataset, protocol.test)
        val_outcomes = _outcomes(dataset, protocol.validation)
        test_outcomes = _outcomes(dataset, protocol.test)

        validation_rows.append(
            evaluate_binary(name, _hard_predictions(model, dataset, protocol.validation), val_outcomes, p_up=val_prob)
        )
        test_rows.append(
            evaluate_binary(name, _hard_predictions(model, dataset, protocol.test), test_outcomes, p_up=test_prob)
        )

        calibrated = calibrate_selective_policy(
            val_prob,
            val_outcomes,
            min_entries=min_validation_entries,
            min_coverage=min_coverage,
        )
        thresholds.append((name, calibrated.policy.confidence_threshold))
        validation_rows.append(
            evaluate_binary(name + " + WAIT", calibrated.policy.apply(val_prob), val_outcomes, p_up=val_prob)
        )
        test_rows.append(
            evaluate_binary(name + " + WAIT", calibrated.policy.apply(test_prob), test_outcomes, p_up=test_prob)
        )

    return BenchmarkResult(
        protocol=protocol,
        validation=tuple(validation_rows),
        test=tuple(test_rows),
        frozen_thresholds=tuple(thresholds),
    )
