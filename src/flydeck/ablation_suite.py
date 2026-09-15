from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .receptive_fields import ReceptiveField
from .scientific_benchmarks import ConfusionMatrix, evaluate_predictions
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class AblationResult:
    variant_name: str
    matrix: ConfusionMatrix
    delta_accuracy: float
    delta_coverage: float
    delta_expectancy: float


def run_agent_over_range(
    agent: FlyVisualPredictionAgent,
    dataset: BNBPredictionDataset,
    start: int,
    end: int,
    context: int = 32,
) -> tuple[Prediction, ...]:
    """Execute fly visual agent over a slice of BNB candles without lookahead."""
    agent.reset()
    predictions: list[Prediction] = []
    for index in range(start, end):
        prices = dataset.closes[max(0, index - context + 1) : index + 1]
        volumes = dataset.volumes[max(0, index - context + 1) : index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=volumes)
        if decision.wait:
            predictions.append(Prediction.WAIT)
        elif decision.up_score > decision.down_score:
            predictions.append(Prediction.UP)
        else:
            predictions.append(Prediction.DOWN)
    return tuple(predictions)


def run_ablation_suite(
    circuit: VisualCircuit,
    dataset: BNBPredictionDataset,
    start: int,
    end: int,
    context: int = 32,
    confidence_threshold: float = 0.20,
    receptive_fields: dict[int, ReceptiveField] | None = None,
) -> tuple[AblationResult, ...]:
    """Evaluate full fly visual agent against mutilation variants removing biological mechanisms."""
    outcomes = tuple(dataset.outcome(index) for index in range(start, end))

    variants = (
        ("Full Model (Bio-Inspired CX+LPTC)", {}),
        ("Ablation: No LPTC Wide-Field Pooling", {"ablate_lptc": True}),
        ("Ablation: No Central Complex Memory", {"ablate_working_memory": True}),
        ("Ablation: No Synaptic Adaptation", {"ablate_adaptation": True}),
        ("Ablation: No Neuromodulatory Arousal", {"ablate_neuromodulation": True}),
        ("Ablation: No Conflict WAIT Engine", {"ablate_conflict_engine": True}),
        ("Ablation: No T4/T5 Motion Detectors", {"ablate_t4_t5": True}),
        ("Ablation: No Spatial Offset (Collapsed)", {"ablate_spatial": True}),
    )

    results: list[AblationResult] = []
    baseline_matrix: ConfusionMatrix | None = None

    for name, kwargs in variants:
        agent = FlyVisualPredictionAgent(
            circuit,
            retina_width=context,
            confidence_threshold=confidence_threshold,
            receptive_fields=receptive_fields,
            **kwargs,
        )
        preds = run_agent_over_range(agent, dataset, start, end, context=context)
        matrix = evaluate_predictions(name, preds, outcomes)

        if baseline_matrix is None:
            baseline_matrix = matrix
            delta_acc = 0.0
            delta_cov = 0.0
            delta_exp = 0.0
        else:
            delta_acc = matrix.accuracy - baseline_matrix.accuracy
            delta_cov = matrix.coverage - baseline_matrix.coverage
            delta_exp = matrix.pancakeswap_expectancy() - baseline_matrix.pancakeswap_expectancy()

        results.append(
            AblationResult(
                variant_name=name,
                matrix=matrix,
                delta_accuracy=delta_acc,
                delta_coverage=delta_cov,
                delta_expectancy=delta_exp,
            )
        )

    return tuple(results)


def format_ablation_table(results: Sequence[AblationResult]) -> str:
    """Format ablation results as a clean markdown/text summary table."""
    headers = f"{'Variant':<42} | {'Acc':>7} | {'Cov':>7} | {'F1':>7} | {'dAcc':>7} | {'dCov':>7} | {'PCS Edge':>8}"
    sep = "-" * len(headers)
    lines = [headers, sep]
    for r in results:
        m = r.matrix
        edge = m.pancakeswap_expectancy()
        d_acc_str = f"{r.delta_accuracy:+.1%}" if r.delta_accuracy != 0.0 else "---"
        d_cov_str = f"{r.delta_coverage:+.1%}" if r.delta_coverage != 0.0 else "---"
        line = (
            f"{r.variant_name:<42} | "
            f"{m.accuracy:>6.1%} | "
            f"{m.coverage:>6.1%} | "
            f"{m.f1_score:>7.3f} | "
            f"{d_acc_str:>7} | "
            f"{d_cov_str:>7} | "
            f"{edge:>+7.2%}"
        )
        lines.append(line)
    return "\n".join(lines)
