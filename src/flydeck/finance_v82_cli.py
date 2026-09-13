from __future__ import annotations

import argparse

from .finance_data import load_ohlcv_csv
from .finance_real import split_real_market, train_real_market_v8
from .finance_real_cli import build_agent
from .finance_v82 import collect_position_aware_diagnostics


HORIZONS = (1, 3, 6, 12, 24)
ACTION_LABELS = ("HOLD", "BUY", "SELL")
STATE_LABELS = ("FLAT", "LONG")


def _print_matrix(title: str, matrix, labels) -> None:
    print(title)
    print("             " + " ".join(f"{label:>6s}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"    {label:5s}: " + " ".join(f"{value:6d}" for value in row))


def _print_returns(title: str, values, labels) -> None:
    print(title)
    print("             " + " ".join(f"{h:>8d}c" for h in HORIZONS))
    for label, row in zip(labels, values):
        print(f"  {label:7s}: " + " ".join(f"{value * 100:8.3f}%" for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck Agent V8.2 position-aware diagnostics")
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_ohlcv_csv(args.data, symbol=args.symbol, interval=args.interval)
    splits = split_real_market(
        dataset,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        context=args.context,
    )
    agent = build_agent(args.seed)
    print("FlyDeck Agent - Real Market Benchmark V8.2 Position-Aware Diagnostics")
    print(f"dataset: {dataset.symbol} {dataset.interval}")
    print(f"rows: {dataset.rows}")
    print(f"train rows: {splits.train.rows} | validation rows: {splits.validation.rows} | test rows: {splits.test.rows}")
    print(f"context candles: {splits.context}\n")

    training = train_real_market_v8(agent, splits.train.candles)
    print("Training on historical train split:")
    print(f"  return: {training.return_pct:.3f}%")
    print(f"  drawdown: {training.drawdown_pct:.3f}%")
    print(f"  trades: {training.trades}")
    print(f"  steps: {training.steps}\n")

    diagnostics = collect_position_aware_diagnostics(agent, splits.test.candles)
    print("V8.2 position-aware diagnostics:")
    print(f"  decisions: {len(diagnostics.records)}")
    print(f"  FLAT decisions: {diagnostics.flat_records}")
    print(f"  LONG decisions: {diagnostics.long_records}")
    print()
    _print_matrix("Action transitions:", diagnostics.action_state_transitions, ACTION_LABELS)
    print()
    _print_matrix("Position state transitions:", diagnostics.position_state_transitions, STATE_LABELS)
    print()

    print("Average exposure change by chosen action:")
    for label, value in zip(ACTION_LABELS, diagnostics.average_exposure_change_by_action):
        print(f"  {label}: {value:+.6f}")
    print()

    print("Exact immediate counterfactual rewards (same pre-action state):")
    for label, value in zip(ACTION_LABELS, diagnostics.average_counterfactual_rewards):
        print(f"  {label}: {value:+.6f}")
    print()

    _print_returns(
        "Realized portfolio return by chosen action:",
        diagnostics.average_realized_portfolio_returns_by_action,
        ACTION_LABELS,
    )
    print()
    _print_returns(
        "Realized portfolio return by position before action:",
        diagnostics.average_realized_portfolio_returns_by_position_state,
        STATE_LABELS,
    )
    print()

    print("Q advantage by position before action:")
    for state, buy, sell in zip(STATE_LABELS, diagnostics.average_buy_advantage_by_position_state, diagnostics.average_sell_advantage_by_position_state):
        print(f"  {state}: BUY-HOLD {buy:+.5f} | SELL-HOLD {sell:+.5f}")

    print("\nSample position/action records:")
    for record in diagnostics.records[:10]:
        cf = ", ".join(f"{label}={value:+.5f}" for label, value in zip(ACTION_LABELS, record.counterfactual_rewards))
        print(
            f"  idx={record.index} {record.position_state_before}->{record.position_state_after} "
            f"action={ACTION_LABELS[record.action]} exposure={record.exposure_change:+.4f} "
            f"Q(B-H)={record.scores[1] - record.scores[0]:+.4f} "
            f"Q(S-H)={record.scores[2] - record.scores[0]:+.4f} cf[{cf}]"
        )


if __name__ == "__main__":
    main()
