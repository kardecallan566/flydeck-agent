from __future__ import annotations

import argparse
from pathlib import Path

from .bnb_prediction_data_runner import load_bnb_5m_csv, run_bnb_prediction_benchmark
from .malecns import MaleCNSCircuit


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck MaleCNS BNB 5-minute prediction")
    parser.add_argument("--circuit", required=True, help="MaleCNS circuit JSON")
    parser.add_argument("--data", help="Historical BNBUSDT 5m CSV")
    parser.add_argument("--download", help="Download latest candles to this CSV")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.20)
    args = parser.parse_args()

    data_path = args.data
    if args.download:
        from .bnb_prediction_data_runner import download_bnb_5m_csv
        data_path = str(download_bnb_5m_csv(args.download, args.limit))
    if not data_path:
        parser.error("use --data or --download")

    data = load_bnb_5m_csv(data_path)
    circuit = MaleCNSCircuit.load(args.circuit)
    train, validation, test = run_bnb_prediction_benchmark(
        data, circuit, confidence_threshold=args.confidence
    )

    print("FlyDeck Agent - MaleCNS BNB 5-Minute Prediction")
    print(f"candles: {data.size}")
    print(f"MaleCNS neurons: {circuit.neurons.__len__()}")
    print(f"MaleCNS edges: {circuit.edges.__len__()}")
    for name, metrics in (("TRAIN", train), ("VALIDATION", validation), ("TEST", test)):
        print(f"\n{name}")
        print(f"rounds: {metrics.rounds}")
        print(f"entered: {metrics.entered}")
        print(f"correct: {metrics.correct}")
        print(f"accuracy: {metrics.accuracy:.3%}")
        print(f"coverage: {metrics.coverage:.3%}")
        print(f"UP/DOWN/WAIT: {metrics.up}/{metrics.down}/{metrics.wait}")


if __name__ == "__main__":
    main()
