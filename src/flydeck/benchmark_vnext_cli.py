from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_vnext import run_vnext_baselines
from .bnb_prediction_data_runner import load_bnb_5m_csv
from .fly_benchmark_vnext import run_vnext_fly
from .pancakeswap_targets import align_pancake_rounds_to_market, load_pancake_rounds_csv
from .statistical_metrics import BinaryEvaluation
from .visual_circuit import VisualCircuit


def _row(metric: BinaryEvaluation) -> str:
    lower, upper = metric.wilson_95
    brier = "-" if metric.brier_score is None else f"{metric.brier_score:.4f}"
    return (
        f"{metric.name:<30} | {metric.entered:>6}/{metric.eligible_rounds:<6} | "
        f"{metric.coverage:>7.1%} | {metric.accuracy:>7.1%} | "
        f"{metric.balanced_accuracy:>7.1%} | {lower:>6.1%}-{upper:<6.1%} | {brier:>7}"
    )


def _print_table(title: str, rows: tuple[BinaryEvaluation, ...]) -> None:
    print(f"\n=== {title} ===")
    print("Model                          | Entered       | Coverage | Accuracy | Bal Acc  | Wilson 95%      | Brier")
    print("-" * 112)
    for row in rows:
        print(_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyDeck vNext leakage-resistant benchmark")
    parser.add_argument("--data", required=True, type=Path, help="Canonical BNBUSDT 5m CSV")
    parser.add_argument(
        "--pancake-rounds",
        type=Path,
        help="PancakeSwap round CSV; switches target to Chainlink lockPrice -> closePrice",
    )
    parser.add_argument(
        "--decision-lead-seconds",
        default=30,
        type=int,
        help="Decision cutoff before PancakeSwap lock; default 30s",
    )
    parser.add_argument("--circuit", type=Path, help="MaleCNS visual circuit JSON")
    parser.add_argument(
        "--fly-ablations",
        action="store_true",
        help="Also run Visual Core, No T4/T5, No MB, No CX and No Predictive Coding",
    )
    parser.add_argument("--fly-confidence", default=0.15, type=float)
    parser.add_argument("--context", default=32, type=int)
    parser.add_argument("--purge", default=1, type=int)
    parser.add_argument("--min-validation-entries", default=100, type=int)
    parser.add_argument("--min-coverage", default=0.10, type=float)
    args = parser.parse_args()

    dataset = load_bnb_5m_csv(args.data)
    alignment = None
    if args.pancake_rounds:
        rounds = load_pancake_rounds_csv(args.pancake_rounds)
        alignment = align_pancake_rounds_to_market(
            dataset,
            rounds,
            decision_lead_seconds=args.decision_lead_seconds,
        )
        dataset = alignment.dataset

    result = run_vnext_baselines(
        dataset,
        context=args.context,
        purge=args.purge,
        min_validation_entries=args.min_validation_entries,
        min_coverage=args.min_coverage,
    )
    p = result.protocol

    validation_rows = result.validation
    test_rows = result.test
    if args.circuit:
        circuit = VisualCircuit.load(args.circuit)
        fly = run_vnext_fly(
            dataset,
            circuit,
            p,
            context=args.context,
            confidence_threshold=args.fly_confidence,
            include_ablations=args.fly_ablations,
        )
        validation_rows = validation_rows + fly.validation
        test_rows = test_rows + fly.test

    print(f"Dataset: {dataset.size} candles")
    print(f"Target: {dataset.target_name}")
    if alignment is not None:
        print(
            "Pancake alignment: "
            f"{alignment.eligible_rounds} eligible, "
            f"{alignment.skipped_invalid} invalid/tie, "
            f"{alignment.skipped_outside_market_history} outside market history, "
            f"{alignment.skipped_duplicate_feature_index} duplicate feature candle"
        )
        print(
            f"Decision cutoff: {args.decision_lead_seconds}s before lock; "
            "only fully closed 5m Binance candles are eligible as features"
        )
    print(f"Train:      [{p.train.start}, {p.train.end}) = {p.train.size}")
    print(f"Validation: [{p.validation.start}, {p.validation.end}) = {p.validation.size}")
    print(f"Test OOS:   [{p.test.start}, {p.test.end}) = {p.test.size}")
    print(f"Purge: {p.purge} prediction index(es) at each boundary")

    _print_table("VALIDATION", validation_rows)
    print("\nFrozen validation-only WAIT thresholds:")
    for name, threshold in result.frozen_thresholds:
        print(f"  {name}: {threshold:.4f}")
    _print_table("TEST OOS - THRESHOLDS FROZEN", test_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
