from __future__ import annotations

import argparse
from pathlib import Path

from .finance_malecns_real import load_btcusdt_dataset, run_malecns_real_benchmark
from .malecns import MaleCNSCircuit
from .malecns_v11 import run_v11_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck MaleCNS real BTCUSDT benchmark")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--data", default="data/real/BTCUSDT_1h.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics", action="store_true", help="run V1.1 training, reservoir and topology diagnostics")
    args = parser.parse_args()

    circuit = MaleCNSCircuit.load(Path(args.circuit))
    dataset = load_btcusdt_dataset(args.data)
    if args.diagnostics:
        diagnostics = run_v11_diagnostics(circuit, dataset.candles, seed=args.seed)
        _print_diagnostics(diagnostics)
        return

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


def _print_diagnostics(diagnostics) -> None:
    print("FlyDeck Agent - MaleCNS V1.1 Diagnostics")
    print("\nTRAINING CHECKPOINTS")
    for item in diagnostics.checkpoints:
        print(f"step={item.step} epsilon={item.epsilon:.4f} "
              f"scores={item.scores[0]:.6f}/{item.scores[1]:.6f}/{item.scores[2]:.6f} "
              f"|td|={item.td_error_mean:.6f} reward={item.reward_mean:.6f}")
    reservoir = diagnostics.reservoir
    print("\nRESERVOIR")
    print(f"mean activity: {reservoir.mean_activity:.6f}")
    print(f"activity std: {reservoir.activity_std:.6f}")
    print(f"active neurons mean: {reservoir.active_neurons_mean:.2f}")
    print(f"unique states: {reservoir.unique_states}")
    print(f"tanh saturation: {reservoir.tanh_saturation_pct:.3f}%")
    readout = diagnostics.readout
    print("\nREADOUT")
    print(f"weights H/B/S: {readout.weights[0]:.6f}/{readout.weights[1]:.6f}/{readout.weights[2]:.6f}")
    print(f"biases H/B/S: {readout.biases[0]:.6f}/{readout.biases[1]:.6f}/{readout.biases[2]:.6f}")
    print(f"score margin to best H/B/S: {readout.score_margin_mean[0]:.6f}/{readout.score_margin_mean[1]:.6f}/{readout.score_margin_mean[2]:.6f}")
    print("\nTOPOLOGY CONTROL")
    print(f"MaleCNS test actions H/B/S: {diagnostics.trained_test_actions[0]}/{diagnostics.trained_test_actions[1]}/{diagnostics.trained_test_actions[2]}")
    print(f"Random topology test actions H/B/S: {diagnostics.random_topology_test_actions[0]}/{diagnostics.random_topology_test_actions[1]}/{diagnostics.random_topology_test_actions[2]}")
    print(f"random topology edges: {diagnostics.random_topology_edges}")


if __name__ == "__main__":
    main()
