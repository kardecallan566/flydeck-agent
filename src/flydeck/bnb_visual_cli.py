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
    parser.add_argument("--exploration", type=float, default=0.30, help="initial exploration rate during survival training")
    parser.add_argument("--min-exploration", type=float, default=0.05)
    parser.add_argument("--wait-streak", type=int, default=8, help="force a directional probe after this many WAITs")
    parser.add_argument("--seed", type=int, default=123)
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
            exploration_rate=args.exploration,
            min_exploration_rate=args.min_exploration,
            max_wait_streak=args.wait_streak,
            seed=args.seed,
        )
        print("FlyDeck visual agent - SURVIVAL TRAINING")
        print(f"rounds: {result.rounds}")
        print(f"deaths: {result.deaths}")
        print(f"lives remaining: {result.lives_remaining}")
        print(f"correct/entered: {result.correct}/{result.entered}")
        print(f"accuracy: {result.correct / result.entered:.3%}" if result.entered else "accuracy: N/A")
        print(f"coverage: {result.entered / result.rounds:.3%}" if result.rounds else "coverage: N/A")
        print(f"exploratory actions: {result.exploratory} (UP/DOWN: {result.exploratory_up}/{result.exploratory_down})")
        print(f"UP/DOWN/WAIT: {result.up}/{result.down}/{result.wait}")
        print(f"death rate per round: {result.deaths / result.rounds:.3%}" if result.rounds else "death rate per round: N/A")
        print("survival rate: deprecated; use accuracy, coverage and death rate per round")
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
        print(f"brier score: {metrics.brier_score:.6f}")
        print(f"expected calibration error: {metrics.expected_calibration_error:.6f}")
        print(f"regimes: {dict(metrics.regimes)}")
        for regime, regime_rounds, entries, accuracy, brier, ece in metrics.regime_metrics:
            print(f"  {regime}: rounds={regime_rounds} entered={entries} accuracy={accuracy:.3%} brier={brier:.6f} ece={ece:.6f}")
        print(f"UP/DOWN/WAIT: {metrics.up}/{metrics.down}/{metrics.wait}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
