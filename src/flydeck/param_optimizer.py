"""Grid search optimizer for FlyDeck Agent hyperparameters.

Searches over key parameters using only the validation split to avoid
test set leakage. The optimization metric is PancakeSwap net expectancy
(accuracy * 0.97 - (1 - accuracy)), which accounts for the 3% treasury fee.

To achieve maximum performance, the visual network forward pass is executed
ONCE to extract sensory traces (directional T4/T5, LPTC wide-field flow,
velocity, ON/OFF balance, coherence), after which thousands of parameter
combinations are evaluated in pure vector/math operations in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .market_retina import BNBMarketRetina
from .receptive_fields import ReceptiveField
from .scientific_benchmarks import ConfusionMatrix, evaluate_predictions
from .visual_agent import MaleCNSVisualSystem
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class ParamSet:
    mutual_inhibition_gamma: float
    trend_memory_beta: float
    confidence_threshold: float
    weight_directional: float
    weight_velocity: float
    weight_on_off_balance: float
    weight_trend: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    params: ParamSet
    matrix: ConfusionMatrix
    expectancy: float


@dataclass(frozen=True, slots=True)
class _StepFeatures:
    raw_up: float
    raw_down: float
    vel_signal: float
    balance_signal: float
    coherence: float
    lptc_net: float
    volatility: float


def _extract_validation_features(
    circuit: VisualCircuit,
    dataset: BNBPredictionDataset,
    val_start: int,
    val_end: int,
    context: int = 32,
    receptive_fields: dict[int, ReceptiveField] | None = None,
) -> tuple[_StepFeatures, ...]:
    """Execute a single forward pass over validation split and extract neural features."""
    retina = BNBMarketRetina(width=context, height=16)
    visual = MaleCNSVisualSystem(
        circuit,
        receptive_fields=receptive_fields,
    )

    l1_inputs = circuit.l1_inputs
    l2_inputs = circuit.l2_inputs
    l1_count = max(1, len(l1_inputs))
    l2_count = max(1, len(l2_inputs))

    features: list[_StepFeatures] = []
    for index in range(val_start, val_end):
        prices = dataset.closes[max(0, index - context + 1) : index + 1]
        volumes = dataset.volumes[max(0, index - context + 1) : index + 1]
        stimulus = retina.encode(prices, volumes=volumes)
        visual.step(stimulus, current_price=prices[-1] if len(prices) > 0 else None)

        # Directional features
        t4 = visual._directional_activity(circuit.t4_outputs)
        t5 = visual._directional_activity(circuit.t5_outputs)
        raw_up = max(0.0, t4[2]) + max(0.0, t5[2])
        raw_down = max(0.0, t4[3]) + max(0.0, t5[3])
        lptc_net = visual.last_lptc_out.vs_net if visual.last_lptc_out else 0.0

        # Velocity & Acceleration
        short_vel = stimulus.short_velocity
        vel_signal = max(-1.0, min(1.0, 0.65 * stimulus.velocity + 0.35 * short_vel + 0.25 * stimulus.acceleration))

        # ON/OFF Circuit Balance
        l1_activity = sum(visual.state[n] for n in l1_inputs)
        l2_activity = sum(visual.state[n] for n in l2_inputs)
        on_mean = l1_activity / l1_count
        off_mean = l2_activity / l2_count
        balance_total = on_mean + off_mean
        balance_signal = (on_mean - off_mean) / balance_total if balance_total > 1e-12 else 0.0

        features.append(
            _StepFeatures(
                raw_up=raw_up,
                raw_down=raw_down,
                vel_signal=vel_signal,
                balance_signal=balance_signal,
                coherence=stimulus.coherence,
                lptc_net=lptc_net,
                volatility=stimulus.volatility_contrast,
            )
        )
    return tuple(features)


def run_grid_search(
    circuit: VisualCircuit,
    dataset: BNBPredictionDataset,
    val_start: int,
    val_end: int,
    context: int = 32,
    receptive_fields: dict[int, ReceptiveField] | None = None,
    gammas: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
    betas: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.40),
    confidences: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
    weight_presets: Sequence[tuple[float, float, float, float]] | None = None,
    top_k: int = 10,
    verbose: bool = True,
) -> tuple[OptimizationResult, ...]:
    """Search over hyperparameters on validation split using cached neural features."""
    if weight_presets is None:
        weight_presets = (
            (0.35, 0.35, 0.20, 0.10),
            (0.20, 0.50, 0.20, 0.10),
            (0.50, 0.20, 0.20, 0.10),
            (0.25, 0.25, 0.35, 0.15),
            (0.10, 0.40, 0.30, 0.20),
            (0.00, 0.50, 0.30, 0.20),
            (0.30, 0.30, 0.30, 0.10),
        )

    outcomes = tuple(dataset.outcome(i) for i in range(val_start, val_end))
    total_combos = len(gammas) * len(betas) * len(confidences) * len(weight_presets)

    if verbose:
        print(f"Extracting neural feature traces over {val_end - val_start} validation rounds...")

    features = _extract_validation_features(
        circuit=circuit,
        dataset=dataset,
        val_start=val_start,
        val_end=val_end,
        context=context,
        receptive_fields=receptive_fields,
    )

    if verbose:
        print(f"Grid search: evaluating {total_combos} combinations on cached traces...")

    results: list[OptimizationResult] = []

    for gamma, beta, conf, (w_d, w_v, w_b, w_t) in product(gammas, betas, confidences, weight_presets):
        w_total = w_d + w_v + w_b + w_t
        inv_w_total = 1.0 / w_total if w_total > 1e-12 else 0.0

        trend_bias = 0.0
        preds: list[Prediction] = []

        for feat in features:
            dir_signal = feat.lptc_net if abs(feat.lptc_net) > 1e-6 else 0.0
            if dir_signal == 0.0:
                d_up = max(0.0, feat.raw_up - gamma * feat.raw_down)
                d_down = max(0.0, feat.raw_down - gamma * feat.raw_up)
                d_tot = d_up + d_down
                dir_signal = (d_up - d_down) / d_tot if d_tot > 1e-12 else 0.0

            instant_bias = dir_signal * 0.5 + feat.vel_signal * 0.5
            trend_bias = (1.0 - beta) * trend_bias + beta * instant_bias
            trend_signal = max(-1.0, min(1.0, trend_bias))

            combined = (
                w_d * dir_signal
                + w_v * feat.vel_signal
                + w_b * feat.balance_signal
                + w_t * trend_signal
            ) * inv_w_total

            eff_up = max(0.0, combined)
            eff_down = max(0.0, -combined)
            confidence = abs(combined)

            # Conflict calculation
            streams = [dir_signal, feat.vel_signal, feat.balance_signal, trend_signal]
            conflicts = []
            for i in range(len(streams)):
                for j in range(i + 1, len(streams)):
                    s_i, s_j = streams[i], streams[j]
                    if (s_i > 0.08 and s_j < -0.08) or (s_i < -0.08 and s_j > 0.08):
                        conflicts.append(abs(s_i - s_j))
            conflict_val = sum(conflicts) / len(conflicts) if conflicts else 0.0

            adaptive_thresh = conf
            if feat.coherence < 0.25:
                adaptive_thresh += (0.25 - feat.coherence) * 0.20
            adaptive_thresh += conflict_val * 0.20

            if confidence < adaptive_thresh:
                preds.append(Prediction.WAIT)
            elif eff_up > eff_down:
                preds.append(Prediction.UP)
            else:
                preds.append(Prediction.DOWN)

        matrix = evaluate_predictions("grid_search", tuple(preds), outcomes)
        expectancy = matrix.pancakeswap_expectancy()

        if matrix.coverage >= 0.12:
            params = ParamSet(
                mutual_inhibition_gamma=gamma,
                trend_memory_beta=beta,
                confidence_threshold=conf,
                weight_directional=w_d,
                weight_velocity=w_v,
                weight_on_off_balance=w_b,
                weight_trend=w_t,
            )
            results.append(OptimizationResult(params=params, matrix=matrix, expectancy=expectancy))

    results.sort(key=lambda r: r.expectancy, reverse=True)

    if verbose:
        print(f"Grid search complete. {len(results)} viable results (coverage >= 12%).")

    return tuple(results[:top_k])


def format_optimization_results(results: Sequence[OptimizationResult]) -> str:
    """Format top-K optimization results as a summary table."""
    lines = [
        f"{'Rank':<5} | {'Gamma':<6} | {'Beta':<6} | {'Conf':<6} | "
        f"{'W_Dir':<6} | {'W_Vel':<6} | {'W_Bal':<6} | {'W_Trd':<6} | "
        f"{'Acc':<7} | {'Cov':<7} | {'PCS Edge':<9}",
        "-" * 95,
    ]
    for i, r in enumerate(results):
        p = r.params
        m = r.matrix
        lines.append(
            f"{i + 1:<5} | {p.mutual_inhibition_gamma:<6.2f} | {p.trend_memory_beta:<6.2f} | "
            f"{p.confidence_threshold:<6.2f} | {p.weight_directional:<6.2f} | "
            f"{p.weight_velocity:<6.2f} | {p.weight_on_off_balance:<6.2f} | "
            f"{p.weight_trend:<6.2f} | {m.accuracy:<6.1%} | {m.coverage:<6.1%} | "
            f"{r.expectancy:>+7.2%}"
        )
    return "\n".join(lines)
