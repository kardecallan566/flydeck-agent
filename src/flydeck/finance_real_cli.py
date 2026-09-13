from __future__ import annotations

import argparse

from .agent import Agent
from .finance_data import download_binance_spot_klines, load_ohlcv_csv
from .finance_encoder import SparseMarketEncoder
from .finance_real import (
    HORIZONS,
    collect_decision_quality,
    evaluate_real_market,
    run_random_baseline,
    split_real_market,
    train_real_market_v8,
)


def build_agent(seed: int = 42) -> Agent:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    return Agent(
        observation_size=encoder.output_size,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=seed,
    )


def _print_transition_matrix(matrix: tuple[tuple[int, ...], ...]) -> None:
    print("  transition matrix (from -> to):")
    print("             HOLD    BUY   SELL")
    for label, row in zip(("HOLD", "BUY ", "SELL"), matrix):
        print(f"    {label}: {row[0]:6d} {row[1]:6d} {row[2]:6d}")


def _print_advantage_buckets(title: str, buckets) -> None:
    print(title)
    print("  bucket         count   1c       3c       6c       12c      24c")
    for bucket in buckets:
        values = " ".join(f"{value * 100:8.3f}%" for value in bucket.average_future_returns)
        print(f"  {bucket.label:12s} {bucket.count:6d} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck Agent real-market data and benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download historical Binance Spot OHLCV data")
    download.add_argument("--symbol", default="BTCUSDT")
    download.add_argument("--interval", default="1h")
    download.add_argument("--start", default="2018-01-01T00:00:00Z")
    download.add_argument("--end", default=None)
    download.add_argument("--output", default="data/real/BTCUSDT_1h.csv")
    download.add_argument("--pause", type=float, default=0.15)

    benchmark = subparsers.add_parser("benchmark", help="train on the chronological train split and evaluate unseen data")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--symbol", default="BTCUSDT")
    benchmark.add_argument("--interval", default="1h")
    benchmark.add_argument("--train-ratio", type=float, default=0.70)
    benchmark.add_argument("--validation-ratio", type=float, default=0.15)
    benchmark.add_argument("--context", type=int, default=24)
    benchmark.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "download":
        dataset = download_binance_spot_klines(
            args.symbol,
            args.interval,
            args.start,
            args.end,
            args.output,
            pause_seconds=args.pause,
        )
        print(f"saved: {args.output}")
        print(f"symbol: {dataset.symbol}")
        print(f"interval: {dataset.interval}")
        print(f"rows: {dataset.rows}")
        print(f"source: {dataset.source}")
        return

    dataset = load_ohlcv_csv(args.data, symbol=args.symbol, interval=args.interval)
    splits = split_real_market(
        dataset,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        context=args.context,
    )
    agent = build_agent(args.seed)

    print("FlyDeck Agent - Real Market Benchmark V8.1 Diagnostics")
    print(f"dataset: {dataset.symbol} {dataset.interval}")
    print(f"rows: {dataset.rows}")
    print(f"train rows: {splits.train.rows} | validation rows: {splits.validation.rows} | test rows: {splits.test.rows}")
    print(f"context candles: {splits.context}")
    print()

    training = train_real_market_v8(agent, splits.train.candles)
    print("Training on historical train split:")
    print(f"  return: {training.return_pct:.3f}%")
    print(f"  drawdown: {training.drawdown_pct:.3f}%")
    print(f"  trades: {training.trades}")
    print(f"  steps: {training.steps}")

    validation = evaluate_real_market(agent, splits.validation.candles)
    print("Validation:")
    print(f"  return: {validation.return_pct:.3f}%")
    print(f"  buy&hold: {validation.buy_hold_return_pct:.3f}%")
    print(f"  drawdown: {validation.drawdown_pct:.3f}%")
    print(f"  trades: {validation.trades}")
    print(f"  HOLD/BUY/SELL: {validation.action_counts}")

    test = evaluate_real_market(agent, splits.test.candles)
    print("Unseen test:")
    print(f"  return: {test.return_pct:.3f}%")
    print(f"  buy&hold: {test.buy_hold_return_pct:.3f}%")
    print(f"  drawdown: {test.drawdown_pct:.3f}%")
    print(f"  trades: {test.trades}")
    print(f"  trade frequency: {test.trade_frequency:.3f}")
    print(f"  HOLD/BUY/SELL: {test.action_counts}")
    print(f"  score means H/B/S: {test.score_means}")
    print(f"  max consecutive BUY: {test.max_buy_streak}")
    print(f"  average holding steps: {test.average_holding_steps:.2f}")

    quality = collect_decision_quality(agent, splits.test.candles)
    print("\nV8.1 decision diagnostics:")
    print("Decision quality (forward close-to-close returns):")
    for action, values in zip(("HOLD", "BUY", "SELL"), quality.average_future_returns_by_action):
        formatted = ", ".join(f"{h}={value * 100:.3f}%" for h, value in zip(HORIZONS, values))
        print(f"  {action}: {formatted}")
    print(f"  BUY-HOLD score advantage: {quality.average_buy_advantage:.5f}")
    print(f"  SELL-HOLD score advantage: {quality.average_sell_advantage:.5f}")
    print(f"  holding durations: {quality.holding_durations}")
    print(f"  completed holdings: {quality.completed_holding_durations}")
    print(f"  average completed holding: {sum(quality.completed_holding_durations) / len(quality.completed_holding_durations):.2f}" if quality.completed_holding_durations else "  average completed holding: 0.00")
    _print_transition_matrix(quality.transition_counts)
    _print_advantage_buckets("\nBUY-HOLD advantage buckets vs future returns:", quality.buy_advantage_buckets)
    _print_advantage_buckets("\nSELL-HOLD advantage buckets vs future returns:", quality.sell_advantage_buckets)

    random_baseline = run_random_baseline(splits.test.candles, seed=args.seed)
    print("\nRandom policy baseline:")
    print(f"  return: {random_baseline.return_pct:.3f}%")
    print(f"  final portfolio: {random_baseline.final_portfolio:.2f}")
    print(f"  drawdown: {random_baseline.drawdown_pct:.3f}%")
    print(f"  trades: {random_baseline.trades}")
    print(f"  trade frequency: {random_baseline.trade_frequency:.3f}")
    print(f"  HOLD/BUY/SELL: {random_baseline.action_counts}")


if __name__ == "__main__":
    main()
