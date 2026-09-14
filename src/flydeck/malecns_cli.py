from __future__ import annotations

import argparse
from pathlib import Path

from .finance_malecns import train_malecns_synthetic
from .malecns import MaleCNSCircuit, download_malecns_data
from .malecns_builder import build_degree_core_circuit


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck MaleCNS crypto experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--output", default="data/malecns/raw")

    build = subparsers.add_parser("build")
    build.add_argument("--annotations", required=True)
    build.add_argument("--weights", required=True)
    build.add_argument("--output", default="data/malecns/degree_core_2048.json")
    build.add_argument("--neurons", type=int, default=2048)
    build.add_argument("--min-synapses", type=int, default=3)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--circuit", required=True)
    synthetic.add_argument("--episodes", type=int, default=20)

    args = parser.parse_args()
    if args.command == "download":
        files = download_malecns_data(args.output)
        for key, path in files.items():
            print(f"{key}: {path}")
        return

    if args.command == "build":
        circuit = build_degree_core_circuit(
            args.annotations,
            args.weights,
            args.output,
            neuron_count=args.neurons,
            min_synapses=args.min_synapses,
        )
        print(f"neurons: {len(circuit.neurons)}")
        print(f"edges: {len(circuit.edges)}")
        print(f"output: {args.output}")
        return

    circuit = MaleCNSCircuit.load(Path(args.circuit))
    result = train_malecns_synthetic(circuit, episodes=args.episodes)
    print("FlyDeck Agent - MaleCNS Crypto V1")
    print(f"neurons: {len(circuit.neurons)}")
    print(f"edges: {len(circuit.edges)}")
    print(f"average return: {result.average_return_pct:.3f}%")
    print(f"best return: {result.best_return_pct:.3f}%")
    print(f"last return: {result.last_return_pct:.3f}%")
    print(f"average drawdown: {result.average_drawdown_pct:.3f}%")
    print(f"trades: {result.total_trades}")
    print(f"trade frequency: {result.trade_frequency:.3f}")
    print(f"HOLD/BUY/SELL: {result.hold_actions}/{result.buy_actions}/{result.sell_actions}")


if __name__ == "__main__":
    main()
