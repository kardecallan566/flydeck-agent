"""New, never-before-seen Binance/optional official-oracle forward experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bnb_prediction_data_runner import load_bnb_5m_csv
from .evolution import fly_feature_cache, load_odds_snapshots, market_features
from .forward_eval import frozen_forward_arm, sha256_file, summarize_frozen_comparison, load_checkpoint
from .pancakeswap_targets import align_pancake_rounds_to_market, load_pancake_rounds_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate already-frozen population checkpoints on NEVER-SEEN candles. Research only."
    )
    parser.add_argument("--data", required=True, type=Path, help="A newly downloaded, distinct CSV")
    prior = parser.add_mutually_exclusive_group(required=True)
    prior.add_argument("--after-file", type=Path, help="Prior inspected recent 5m CSV, for strict timestamp exclusion")
    prior.add_argument("--after-meta", type=Path, help="Prior inspected recent download metadata JSON")
    parser.add_argument("--with-checkpoint", required=True, type=Path)
    parser.add_argument("--without-checkpoint", required=True, type=Path)
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--fly-cache", type=Path, default=Path("data/cache/fly_forward_frozen_v3.npz"))
    parser.add_argument("--rebuild-fly-cache", action="store_true")
    parser.add_argument("--no-masked-signal", action="store_true", help="Skip the same-weights zero-Fly sensitivity check")
    parser.add_argument("--trace-all", action="store_true", help="Log decisions from all 100+ agents; larger CSVs")
    parser.add_argument("--pancake-rounds", type=Path, help="Optional official settled epoch CSV")
    parser.add_argument("--odds-snapshots", type=Path, help="Optional pre-lock quotes; only meaningful with official rounds")
    parser.add_argument("--decision-lead-seconds", type=int, default=30)
    parser.add_argument("--resume-root", type=Path, help="Previous forward run root for persistent paper equity")
    parser.add_argument(
        "--cumulative-manifest", type=Path,
        help="Explicitly replay an append-only already-inspected forward stream. "
             "Requires flydeck-append-candles manifest; never use --resume-root.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cumulative_manifest and args.resume_root:
        parser.error("--cumulative-manifest replays ALL prior candles; never combine with --resume-root")
    if args.odds_snapshots and not args.pancake_rounds:
        parser.error("--odds-snapshots requires --pancake-rounds")
    if args.after_file:
        last_seen = load_bnb_5m_csv(args.after_file).timestamps[-1]
    else:
        meta = json.loads(args.after_meta.read_text(encoding="utf-8"))
        if meta.get("symbol") != "BNBUSDT" or meta.get("interval") != "5m":
            parser.error("--after-meta must describe BNBUSDT 5m data")
        last_seen = int(meta["last_open_ms"])

    fly_cp, _ = load_checkpoint(args.with_checkpoint, expect_fly=True)
    base_cp, _ = load_checkpoint(args.without_checkpoint, expect_fly=False)
    if (fly_cp["last_historical_candle_open_ms"] != base_cp["last_historical_candle_open_ms"]
        or fly_cp["source_target"] != base_cp["source_target"]
        or fly_cp["settings"] != base_cp["settings"]
        or (fly_cp.get("source_candles_sha256") and base_cp.get("source_candles_sha256")
            and fly_cp["source_candles_sha256"] != base_cp["source_candles_sha256"])):
        parser.error("frozen arms must use the same history, target and settings")
    if (fly_cp.get("circuit_sha256")
        and fly_cp["circuit_sha256"] != sha256_file(args.circuit)):
        parser.error("MaleCNS circuit differs from the one used in frozen training")

    if args.cumulative_manifest:
        manifest = json.loads(args.cumulative_manifest.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1
            or manifest.get("cumulative_already_inspected") is not True
            or manifest.get("cumulative_sha256") != sha256_file(args.data)):
            parser.error("invalid cumulative manifest or CSV hash mismatch")
    else:
        manifest = None

    data = load_bnb_5m_csv(args.data)
    if manifest is not None:
        if (data.size != int(manifest["total_rows"])
            or data.timestamps[0] != int(manifest["first_open_ms"])
            or data.timestamps[-1] != int(manifest["last_open_ms"])
            or int(manifest["first_new_open_ms"]) <= last_seen):
            parser.error("cumulative dataset provenance/timestamps inconsistent with prior inspected file")
    alignment = None
    if args.pancake_rounds:
        alignment = align_pancake_rounds_to_market(
            data, load_pancake_rounds_csv(args.pancake_rounds),
            decision_lead_seconds=args.decision_lead_seconds,
        )
        data = alignment.dataset
    odds = load_odds_snapshots(args.odds_snapshots, alignment) if args.odds_snapshots else None
    signal = fly_feature_cache(
        data, market_file=args.data, circuit_file=args.circuit,
        cache_file=args.fly_cache, rebuild=args.rebuild_fly_cache,
    )
    sha = sha256_file(args.data)
    treatment = frozen_forward_arm(
        checkpoint_file=args.with_checkpoint, data=data,
        features=market_features(data, signal),
        output=args.output / "with_fly",
        after_timestamp_ms=last_seen, source_csv_sha256=sha,
        expect_fly=True, name="with_fly", alignment=alignment,
        odds_by_epoch=odds, trace_all=args.trace_all,
        cumulative_inspected=manifest is not None,
        resume_state=(
            args.resume_root / "with_fly" / "forward_state.json"
            if args.resume_root else None
        ),
    )
    control = frozen_forward_arm(
        checkpoint_file=args.without_checkpoint, data=data,
        features=market_features(data),
        output=args.output / "without_fly",
        after_timestamp_ms=last_seen, source_csv_sha256=sha,
        expect_fly=False, name="without_fly", alignment=alignment,
        odds_by_epoch=odds, trace_all=args.trace_all,
        cumulative_inspected=manifest is not None,
        resume_state=(
            args.resume_root / "without_fly" / "forward_state.json"
            if args.resume_root else None
        ),
    )
    masked = None
    if not args.no_masked_signal:
        masked = frozen_forward_arm(
            checkpoint_file=args.with_checkpoint, data=data,
            features=market_features(data),  # same exact weights, neural feature forced to zero
            output=args.output / "with_fly_signal_masked",
            after_timestamp_ms=last_seen, source_csv_sha256=sha,
            expect_fly=True, name="with_fly_signal_masked",
            alignment=alignment, odds_by_epoch=odds, trace_all=args.trace_all,
            cumulative_inspected=manifest is not None,
            resume_state=(
                args.resume_root / "with_fly_signal_masked" / "forward_state.json"
                if args.resume_root else None
            ),
        )
    report = summarize_frozen_comparison(treatment, control, masked, args.output)
    print("Frozen prospective research finished:", args.output)
    for label in ("with_fly", "without_fly"):
        row = report[label]["full_cohort"]
        print(
            f"{label}: positive={row['positive_this_window']}/{row['agents']} "
            f"median window return={row['median_window_return_pct']:.3f}% "
            f"median accuracy ACTIVE="
            f"{row['median_accuracy_active_only']:.3%} "
            f"({row['agents_without_entries']} agents entered zero times) "
            f"median coverage={row['median_coverage']:.1%}"
        )
    if manifest is not None:
        print("CUMULATIVE REPLAY: old inspected candles are INCLUDED. "
              "Not a new independent holdout. Full model replay; no double-counted capital.")
    print("Review frozen_comparison.json and each arm's preselected_decisions.csv.")
    print("No model was trained or promoted using this holdout. NO REAL BETS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
