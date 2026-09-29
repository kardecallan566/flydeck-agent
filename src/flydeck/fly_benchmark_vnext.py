from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .probability_calibration import fit_platt_calibrator
from .receptive_fields import ReceptiveField, infer_receptive_fields
from .selective_policy import calibrate_selective_policy
from .statistical_metrics import BinaryEvaluation, evaluate_binary
from .validation_protocol import ChronologicalProtocol, IndexRange
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class FlyBenchmarkResult:
    validation: tuple[BinaryEvaluation, ...]
    test: tuple[BinaryEvaluation, ...]
    frozen_thresholds: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class _AgentTrace:
    predictions: tuple[Prediction, ...]
    forced_predictions: tuple[Prediction, ...]
    p_up: tuple[float, ...]


def run_vnext_fly(
    dataset: BNBPredictionDataset,
    circuit: VisualCircuit,
    protocol: ChronologicalProtocol,
    *,
    context: int = 32,
    confidence_threshold: float = 0.15,
    include_ablations: bool = False,
    receptive_fields: dict[int, ReceptiveField] | None = None,
    min_validation_entries: int = 100,
    min_coverage: float = 0.10,
) -> FlyBenchmarkResult:
    """Run FlyDeck on frozen boundaries and add held-out calibrated selection.

    Plasticity is enabled only in training. Validation/test preserve learned
    weights but disable learning. For each variant, the first half of validation
    fits Platt scaling and the second half selects asymmetric UP/DOWN WAIT
    thresholds. Both are frozen before OOS evaluation.
    """
    fields = receptive_fields or infer_receptive_fields(circuit, iterations=6)
    variants: list[tuple[str, dict[str, bool]]] = [("FlyDeck Full", {})]
    if include_ablations:
        variants.extend(
            (
                ("FlyDeck Visual Core", {
                    "ablate_adaptation": True,
                    "ablate_working_memory": True,
                    "ablate_neuromodulation": True,
                    "ablate_conflict_engine": True,
                    "ablate_mushroom_body": True,
                    "ablate_predictive_coding": True,
                    "ablate_attention": True,
                    "ablate_giant_fiber": True,
                    "ablate_metabolic": True,
                }),
                ("FlyDeck No T4/T5", {"ablate_t4_t5": True}),
                ("FlyDeck No MB", {"ablate_mushroom_body": True}),
                ("FlyDeck No CX", {"ablate_working_memory": True}),
                ("FlyDeck No Predictive Coding", {"ablate_predictive_coding": True}),
            )
        )

    validation_rows: list[BinaryEvaluation] = []
    test_rows: list[BinaryEvaluation] = []
    thresholds: list[tuple[str, float, float]] = []
    validation_outcomes = _outcomes(dataset, protocol.validation)
    test_outcomes = _outcomes(dataset, protocol.test)

    for name, kwargs in variants:
        agent = FlyVisualPredictionAgent(
            circuit,
            retina_width=context,
            confidence_threshold=confidence_threshold,
            receptive_fields=fields,
            **kwargs,
        )
        _run_trace(agent, dataset, protocol.train, context=context, collect=False)
        agent.set_learning(False)

        agent.reset(preserve_learning=True)
        validation_trace = _run_trace(agent, dataset, protocol.validation, context=context, collect=True)
        validation_rows.append(
            evaluate_binary(
                name,
                validation_trace.predictions,
                validation_outcomes,
                p_up=validation_trace.p_up,
            )
        )

        agent.reset(preserve_learning=True)
        test_trace = _run_trace(agent, dataset, protocol.test, context=context, collect=True)
        test_rows.append(
            evaluate_binary(
                name,
                test_trace.predictions,
                test_outcomes,
                p_up=test_trace.p_up,
            )
        )

        validation_rows.append(
            evaluate_binary(
                name + " - No WAIT",
                validation_trace.forced_predictions,
                validation_outcomes,
                p_up=validation_trace.p_up,
            )
        )
        test_rows.append(
            evaluate_binary(
                name + " - No WAIT",
                test_trace.forced_predictions,
                test_outcomes,
                p_up=test_trace.p_up,
            )
        )

        split = max(1, len(validation_trace.p_up) // 2)
        calibrator = fit_platt_calibrator(
            validation_trace.p_up[:split],
            validation_outcomes[:split],
        )
        val_prob = calibrator.apply(validation_trace.p_up)
        test_prob = calibrator.apply(test_trace.p_up)
        try:
            selective = calibrate_selective_policy(
                val_prob[split:],
                validation_outcomes[split:],
                min_entries=min_validation_entries,
                min_coverage=min_coverage,
                asymmetric=True,
            )
        except ValueError:
            continue

        thresholds.append((name, selective.policy.up_threshold, selective.policy.down_threshold))
        validation_rows.append(
            evaluate_binary(
                name + " + CAL + WAIT",
                selective.policy.apply(val_prob),
                validation_outcomes,
                p_up=val_prob,
            )
        )
        test_rows.append(
            evaluate_binary(
                name + " + CAL + WAIT",
                selective.policy.apply(test_prob),
                test_outcomes,
                p_up=test_prob,
            )
        )

    return FlyBenchmarkResult(
        validation=tuple(validation_rows),
        test=tuple(test_rows),
        frozen_thresholds=tuple(thresholds),
    )


def _run_trace(
    agent: FlyVisualPredictionAgent,
    dataset: BNBPredictionDataset,
    span: IndexRange,
    *,
    context: int,
    collect: bool,
) -> _AgentTrace:
    predictions: list[Prediction] = []
    forced: list[Prediction] = []
    probabilities: list[float] = []

    for index in range(span.start, span.end):
        start = max(0, index - context + 1)
        prices = dataset.closes[start : index + 1]
        volumes = dataset.volumes[start : index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=volumes)
        if not collect:
            continue

        directional = Prediction.UP if decision.up_score >= decision.down_score else Prediction.DOWN
        prediction = Prediction.WAIT if decision.wait else directional
        directional_total = max(1e-12, decision.p_up + decision.p_down)
        p_up = min(1.0, max(0.0, decision.p_up / directional_total))
        predictions.append(prediction)
        forced.append(directional)
        probabilities.append(p_up)

    return _AgentTrace(
        predictions=tuple(predictions),
        forced_predictions=tuple(forced),
        p_up=tuple(probabilities),
    )


def _outcomes(dataset: BNBPredictionDataset, span: IndexRange) -> tuple[Prediction, ...]:
    return tuple(dataset.outcome(index) for index in range(span.start, span.end))
