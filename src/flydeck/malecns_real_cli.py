from __future__ import annotations

import argparse
from pathlib import Path

from .finance_malecns_real import load_btcusdt_dataset, run_malecns_real_benchmark
from .malecns import MaleCNSCircuit


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck MaleCNS real BTCUSDT benchmark")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--data", default="data/real/BTCUSDT_1h.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    circuit = MaleCNSCircuit.load(Path(args.circuit))
    dataset = load_btcusdt_dataset(args.data)
    benchmark = run_malecns_real_benchmark(circuit, dataset, seed=args.seed)

    print("FlyDeck Agent - MaleCNS Crypto V1 Real Benchmark")
    print(f"dataset: {dataset.symbol} {dataset.interval}")
    print(f"rows: {dataset.rows}")
    print(f"circuit neurons: {len(circuit.neurons)}")
    print(f"circuit edges: {len(circuit.edges)}")
    _print_result("TRAIN", benchmark.train)
    _print_result("VALIDATION", benchmark.validation)
    _print_result("TEST", benchmark.test)
    _print_result("RANDOM TEST", benchmark.random_test)


def _print_result(name: str, result) -> None:
    hold, buy, sell = result.action_counts
    print(f"\n{name}")
    print(f"return: {result.return_pct:.3f}%")
    print(f"buy & hold: {result.buy_hold_return_pct:.3f}%")
    print(f"final portfolio: {result.final_portfolio:.2f}")
    print(f"drawdown: {result.drawdown_pct:.3f}%")
    print(f"trades: {result.trades}")
    print(f"steps: {result.steps}")
    print(f"HOLD/BUY/SELL: {hold}/{buy}/{sell}")
    print(f"score means H/B/S: {result.score_means[0]:.6f}/{result.score_means[1]:.6f}/{result.score_means[2]:.6f}")


if __name__ == "__main__":
    main()
