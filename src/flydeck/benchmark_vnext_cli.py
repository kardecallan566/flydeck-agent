from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_vnext import run_vnext_baselines
from .bnb_prediction_data_runner import load_bnb_5m_csv
from .statistical_metrics import BinaryEvaluation


def _row(metric: BinaryEvaluation) -> str:
    lower, upper = metric.wilson_95
    brier = "-" if metric.brier_score is None else f"{metric.brier_score:.4f}"
    return (
        f"{metric.name:<26} | {metric.entered:>6}/{metric.eligible_rounds:<6} | "
        f"{metric.coverage:>7.1%} | {metric.accuracy:>7.1%} | "
        f"{metric.balanced_accuracy:>7.1%} | {lower:>6.1%}-{upper:<6.1%} | {brier:>7}"
    )


def _print_table(title: str, rows: tuple[BinaryEvaluation, ...]) -> None:
    print(f"\n=== {title} ===")
    print("Model                      | Entered       | Coverage | Accuracy | Bal Acc  | Wilson 95%      | Brier")
    print("-" * 108)
    for row in rows:
        print(_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyDeck vNext leakage-resistant baseline benchmark")
    parser.add_argument("--data", required=True, type=Path, help="Canonical BNBUSDT 5m CSV")
    parser.add_argument("--context", default=32, type=int)
    parser.add_argument("--purge", default=1, type=int)
    parser.add_argument("--min-validation-entries", default=100, type=int)
    parser.add_argument("--min-coverage", default=0.10, type=float)
    args = parser.parse_args()

    dataset = load_bnb_5m_csv(args.data)
    result = run_vnext_baselines(
        dataset,
        context=args.context,
        purge=args.purge,
        min_validation_entries=args.min_validation_entries,
        min_coverage=args.min_coverage,
    )
    p = result.protocol
    print(f"Dataset: {dataset.size} candles")
    print(f"Train:      [{p.train.start}, {p.train.end}) = {p.train.size}")
    print(f"Validation: [{p.validation.start}, {p.validation.end}) = {p.validation.size}")
    print(f"Test OOS:   [{p.test.start}, {p.test.end}) = {p.test.size}")
    print(f"Purge: {p.purge} prediction index(es) at each boundary")

    _print_table("VALIDATION", result.validation)
    print("\nFrozen validation-only WAIT thresholds:")
    for name, threshold in result.frozen_thresholds:
        print(f"  {name}: {threshold:.4f}")
    _print_table("TEST OOS - THRESHOLDS FROZEN", result.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
