from __future__ import annotations

import argparse

from .agent import Agent
from .finance import CryptoTradingEnvironment
from .finance_data import load_ohlcv_csv
from .finance_encoder import SparseMarketEncoder
from .finance_real import collect_decision_quality, run_random_baseline, split_real_market
from .finance_v82 import collect_position_aware_diagnostics
from .finance_v83 import train_real_market_v83, train_synthetic_crypto_v83


def _agent(seed: int = 42) -> Agent:
    environment = CryptoTradingEnvironment(
        __import__("flydeck.finance", fromlist=["SyntheticCryptoMarket"]).SyntheticCryptoMarket(
            length=256, seed=100
        ).generate()
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    return Agent(
        observation_size=encoder.output_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=seed,
    )


def _print_position_diagnostics(diagnostics) -> None:
    labels = ("HOLD", "BUY", "SELL")
    print("\nV8.3 position-aware diagnostics:")
    print(f"  decisions: {len(diagnostics.records)}")
    print(f"  FLAT decisions: {diagnostics.flat_records}")
    print(f"  LONG decisions: {diagnostics.long_records}")
    print("\nAction transitions:")
    print("               HOLD    BUY   SELL")
    for index, row in enumerate(diagnostics.action_state_transitions):
        print(f"    {labels[index]:<5}: {row[0]:6d} {row[1]:6d} {row[2]:6d}")
    print("\nPosition state transitions:")
    print("               FLAT   LONG")
    print(f"    FLAT : {diagnostics.position_state_transitions[0][0]:6d} {diagnostics.position_state_transitions[0][1]:7d}")
    print(f"    LONG : {diagnostics.position_state_transitions[1][0]:6d} {diagnostics.position_state_transitions[1][1]:7d}")
    print("\nAverage exposure change by chosen action:")
    for index, label in enumerate(labels):
        print(f"  {label}: {diagnostics.average_exposure_change_by_action[index]:+.6f}")
    print("\nExact immediate counterfactual rewards (same pre-action state):")
    for index, label in enumerate(labels):
        print(f"  {label}: {diagnostics.average_counterfactual_rewards[index]:+.6f}")
    print("\nRealized portfolio return by chosen action:")
    print("                    1c        3c        6c       12c       24c")
    for index, label in enumerate(labels):
        values = diagnostics.average_realized_portfolio_returns_by_action[index]
        print(f"  {label:<6}: " + " ".join(f"{value:+.3%}" for value in values))
    print("\nRealized portfolio return by position before action:")
    print("                    1c        3c        6c       12c       24c")
    for index, label in enumerate(("FLAT", "LONG")):
        values = diagnostics.average_realized_portfolio_returns_by_position_state[index]
        print(f"  {label:<6}: " + " ".join(f"{value:+.3%}" for value in values))
    print("\nQ advantage by position before action:")
    print(f"  FLAT: BUY-HOLD {diagnostics.average_buy_advantage_by_position_state[0]:+.5f} | SELL-HOLD {diagnostics.average_sell_advantage_by_position_state[0]:+.5f}")
    print(f"  LONG: BUY-HOLD {diagnostics.average_buy_advantage_by_position_state[1]:+.5f} | SELL-HOLD {diagnostics.average_sell_advantage_by_position_state[1]:+.5f}")
    print("\nSample position/action records:")
    for record in diagnostics.records[:10]:
        cf = record.counterfactual_rewards
        print(
            f"  idx={record.index} {record.position_state_before}->{record.position_state_after} "
            f"action={labels[record.action]} exposure={record.exposure_change:+.4f} "
            f"Q(B-H)={record.scores[1]-record.scores[0]:+.4f} "
            f"Q(S-H)={record.scores[2]-record.scores[0]:+.4f} "
            f"cf[HOLD={cf[0]:+.5f}, BUY={cf[1]:+.5f}, SELL={cf[2]:+.5f}]"
        )


def benchmark(path: str, symbol: str, interval: str, train_ratio: float, validation_ratio: float, context: int, seed: int) -> None:
    dataset = load_ohlcv_csv(path, symbol=symbol, interval=interval)
    splits = split_real_market(dataset, train_ratio=train_ratio, validation_ratio=validation_ratio, context=context)
    agent = _agent(seed)
    print("FlyDeck Agent - Real Market Benchmark V8.3 All-Action TD")
    print(f"dataset: {dataset.symbol} {dataset.interval}")
    print(f"rows: {dataset.rows}")
    print(f"train rows: {splits.train.rows} | validation rows: {splits.validation.rows} | test rows: {splits.test.rows}")
    print(f"context candles: {context}")
    print("\nTraining with independent TD targets for HOLD / BUY / SELL:")
    result = train_real_market_v83(agent, splits.train.candles)
    print(f"  return: {result[0]:.3f}%")
    print(f"  drawdown: {result[1]:.3f}%")
    print(f"  trades: {result[2]}")
    print(f"  steps: {result[3]}")
    print(f"  HOLD/BUY/SELL: {result[4]}")
    print(f"  average TD error: {result[5]:.5f}")

    validation = collect_position_aware_diagnostics(agent, splits.validation.candles)
    test = collect_position_aware_diagnostics(agent, splits.test.candles)
    print("\nValidation position-aware diagnostics:")
    print(f"  decisions: {len(validation.records)} | FLAT: {validation.flat_records} | LONG: {validation.long_records}")
    print(f"  average counterfactual rewards: {tuple(round(v, 6) for v in validation.average_counterfactual_rewards)}")
    print("\nUnseen test position-aware diagnostics:")
    _print_position_diagnostics(test)

    random_result = run_random_baseline(splits.test.candles)
    print("\nRandom policy baseline:")
    print(f"  return: {random_result.return_pct:.3f}%")
    print(f"  final portfolio: {random_result.final_portfolio:.2f}")
    print(f"  drawdown: {random_result.drawdown_pct:.3f}%")
    print(f"  trades: {random_result.trades}")
    print(f"  trade frequency: {random_result.trade_frequency:.3f}")
    print(f"  HOLD/BUY/SELL: {random_result.action_counts}")


def synthetic() -> None:
    environment = CryptoTradingEnvironment(
        __import__("flydeck.finance", fromlist=["SyntheticCryptoMarket"]).SyntheticCryptoMarket(length=256, seed=100).generate(),
        max_steps=200,
    )
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    agent = Agent(
        observation_size=encoder.output_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )
    result = train_synthetic_crypto_v83(agent)
    print("FlyDeck Agent - Synthetic V8.3 All-Action TD")
    print(f"episodes: {result.episodes}")
    print(f"average return: {result.average_return_pct:.3f}%")
    print(f"best return: {result.best_return_pct:.3f}%")
    print(f"last return: {result.last_return_pct:.3f}%")
    print(f"average drawdown: {result.average_drawdown_pct:.3f}%")
    print(f"trades: {result.total_trades}")
    print(f"trade frequency: {result.trade_frequency:.3f}")
    print(f"HOLD/BUY/SELL: {result.hold_actions}/{result.buy_actions}/{result.sell_actions}")
    print(f"average TD error: {result.average_prediction_error:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck Agent V8.3 all-action TD benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.set_defaults(handler=lambda args: synthetic())

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--data", required=True)
    benchmark_parser.add_argument("--symbol", default="BTCUSDT")
    benchmark_parser.add_argument("--interval", default="1h")
    benchmark_parser.add_argument("--train-ratio", type=float, default=0.70)
    benchmark_parser.add_argument("--validation-ratio", type=float, default=0.15)
    benchmark_parser.add_argument("--context", type=int, default=24)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.set_defaults(handler=lambda args: benchmark(args.data, args.symbol, args.interval, args.train_ratio, args.validation_ratio, args.context, args.seed))

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
