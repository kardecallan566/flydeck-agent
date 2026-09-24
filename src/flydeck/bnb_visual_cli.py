from __future__ import annotations

import argparse
from pathlib import Path

from .bnb_prediction_data_runner import load_bnb_5m_csv
from .bnb_visual_runner import run_visual_benchmark
from .survival_training import train_visual_survival
from .visual_circuit import VisualCircuit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BNB through the MaleCNS visual agent")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--survival", action="store_true", help="train with finite lives and causal deaths")
    parser.add_argument("--lives", type=int, default=3, help="lives per survival episode")
    parser.add_argument("--rounds", type=int, default=None, help="maximum chronological training rounds")
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()

    data = load_bnb_5m_csv(args.data)
    circuit = VisualCircuit.load(args.circuit)
    if args.survival:
        _agent, result = train_visual_survival(
            data,
            circuit,
            initial_lives=args.lives,
            max_rounds=args.rounds,
            confidence_threshold=args.confidence,
        )
        print("FlyDeck visual agent - SURVIVAL TRAINING")
        print(f"rounds: {result.rounds}")
        print(f"deaths: {result.deaths}")
        print(f"lives remaining: {result.lives_remaining}")
        print(f"correct/entered: {result.correct}/{result.entered}")
        print(f"UP/DOWN/WAIT: {result.up}/{result.down}/{result.wait}")
        print(f"survival rate: {result.survival_rate:.3%}")
        return 0
    train, validation, test = run_visual_benchmark(data, circuit)
    print(f"visual neurons: {len(circuit.neurons)}")
    print(f"visual edges: {len(circuit.edges)}")
    for name, metrics in (("TRAIN", train), ("VALIDATION", validation), ("TEST", test)):
        print(f"\n{name}")
        print(f"rounds: {metrics.rounds}")
        print(f"entered: {metrics.entered}")
        print(f"correct: {metrics.correct}")
        print(f"accuracy: {metrics.accuracy:.3%}")
        print(f"coverage: {metrics.coverage:.3%}")
        print(f"UP/DOWN/WAIT: {metrics.up}/{metrics.down}/{metrics.wait}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
