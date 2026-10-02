"""CLI for the FlyDeck population evolution research runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from .bnb_prediction_data_runner import load_bnb_5m_csv
from .evolution import (
    EvolutionSettings, fly_feature_cache, load_odds_snapshots, run_evolution, _hash,
)
from .evolution_comparison import write_ablation_comparison
from .pancakeswap_targets import align_pancake_rounds_to_market, load_pancake_rounds_csv


def _dataset(args, data_path: Path, rounds_path: Path | None):
    dataset = load_bnb_5m_csv(data_path)
    alignment = None
    if rounds_path:
        alignment = align_pancake_rounds_to_market(
            dataset, load_pancake_rounds_csv(rounds_path),
            decision_lead_seconds=args.lead_seconds,
        )
        dataset = alignment.dataset
    return dataset, alignment


def main() -> int:
    p = argparse.ArgumentParser(
        description="100–300 lightweight agents, chronological evolution and sealed recent holdout. RESEARCH ONLY."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--population", type=int, choices=(100, 200, 300), default=100)
    p.add_argument("--output", type=Path, default=Path("data/evolution/run"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-no-fly", action="store_true", help="Run matched seed/timestamps with and without frozen MaleCNS features; doubles cheap population pass, not neural cache")
    p.add_argument("--block-size", type=int, default=10000)
    p.add_argument("--circuit", type=Path, help="Optional circuit; compute one shared frozen Fly signal cache")
    p.add_argument("--fly-cache", type=Path, default=Path("data/cache/fly_shared_features.npz"))
    p.add_argument("--rebuild-fly-cache", action="store_true")
    p.add_argument("--pancake-rounds", type=Path, help="Historical Chainlink outcomes from flydeck-pancake-rounds")
    p.add_argument("--odds-snapshots", type=Path, help="Pre-decision epoch,timestamp_ms,gross_up,gross_down CSV")
    p.add_argument("--lead-seconds", type=int, default=30)
    p.add_argument("--recent-data", type=Path, help="Chronologically newer 5m CSV for adaptation plus sealed recent holdout")
    p.add_argument("--recent-rounds", type=Path)
    p.add_argument("--recent-odds-snapshots", type=Path)
    p.add_argument("--recent-fly-cache", type=Path, default=Path("data/cache/recent_fly_shared_features.npz"))
    p.add_argument("--recent-holdout", type=int, default=None, help="Number of most recent candles to keep sealed. Default: ALL --recent-data candles, without recent adaptation")
    p.add_argument("--min-entries", type=int, default=80)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--scenario-gross-odds", type=float, default=2.0)
    p.add_argument("--scenario-fee", type=float, default=0.03)
    p.add_argument("--gas-fraction-of-stake", type=float, default=0.0)
    args = p.parse_args()
    if args.compare_no_fly and not args.circuit:
        p.error("--compare-no-fly requires --circuit")
    if args.odds_snapshots and not args.pancake_rounds:
        p.error("--odds-snapshots requires --pancake-rounds")
    if args.recent_rounds and not args.recent_data:
        p.error("--recent-rounds requires --recent-data")
    if args.recent_odds_snapshots and not args.recent_rounds:
        p.error("--recent-odds-snapshots requires --recent-rounds")
    if args.recent_data and args.circuit and not args.recent_fly_cache:
        p.error("--recent-fly-cache required if circuit is present")

    data, alignment = _dataset(args, args.data, args.pancake_rounds)
    fly = None
    if args.circuit:
        fly = fly_feature_cache(
            data, market_file=args.data, circuit_file=args.circuit,
            cache_file=args.fly_cache, rebuild=args.rebuild_fly_cache,
        )
    odds = load_odds_snapshots(args.odds_snapshots, alignment) if args.odds_snapshots else None

    recent = recent_alignment = recent_fly = recent_odds = None
    if args.recent_data:
        recent, recent_alignment = _dataset(args, args.recent_data, args.recent_rounds)
        if args.circuit:
            recent_fly = fly_feature_cache(
                recent, market_file=args.recent_data, circuit_file=args.circuit,
                cache_file=args.recent_fly_cache, rebuild=args.rebuild_fly_cache,
            )
        if args.recent_odds_snapshots:
            recent_odds = load_odds_snapshots(args.recent_odds_snapshots, recent_alignment)

    settings = EvolutionSettings(
        population=args.population, seed=args.seed, block_size=args.block_size,
        min_entries=args.min_entries, min_coverage=args.min_coverage,
        scenario_gross_odds=args.scenario_gross_odds, scenario_fee=args.scenario_fee,
        gas_fraction_of_stake=args.gas_fraction_of_stake,
    )
    sealed_count = args.recent_holdout if args.recent_holdout is not None else (
        recent.size if recent is not None else 2016
    )
    def execute(out, signal, recent_signal):
        return run_evolution(
            data, settings, output=out, alignment=alignment, odds_by_epoch=odds,
            fly_signal=signal, recent=recent, recent_alignment=recent_alignment,
            recent_fly_signal=recent_signal, recent_odds_by_epoch=recent_odds,
            recent_holdout=sealed_count,
            source_candles_sha256=_hash(args.data),
            circuit_sha256=_hash(args.circuit) if signal is not None else None,
        )

    if args.compare_no_fly:
        print("CONTROLLED EXPERIMENT: shared MaleCNS versus OHLCV-only; "
              "same seed, block splits, risk settings and historical data.")
        execute(args.output / "with_fly", fly, recent_fly)
        execute(args.output / "without_fly", None, None)
        report = write_ablation_comparison(args.output)
        print("ABLATED COMPARISON SAVED:", args.output / "ablation_report.json")
        print("Holdout results are DESCRIPTIVE; no agent is live-approved.")
        print("Audit median-equity difference (with - without):",
              report["delta_with_minus_without"]["all_historical_audit"]["median_equity"])
    else:
        result = execute(args.output, fly, recent_fly)
        print(f"Finished: {result['population']} agents | {args.output}")
        print("Review summary.json, final_population.csv and all_block_results.csv.")
    print("Research only: no real-money execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
