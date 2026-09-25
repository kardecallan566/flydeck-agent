from __future__ import annotations

import argparse
from pathlib import Path

from .bnb_prediction_data_runner import load_bnb_5m_csv
from .bnb_visual_runner import run_visual_benchmark
from .crypto_event_runner import run_crypto_event_benchmark
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
    parser.add_argument("--crypto-event", action="store_true", help="evaluate continuous multi-horizon crypto policy")
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    args = parser.parse_args()

    data = load_bnb_5m_csv(args.data)
    circuit = VisualCircuit.load(args.circuit)
    if args.crypto_event:
        from .crypto_event_policy import CryptoEventConfig
        metrics = run_crypto_event_benchmark(
            data, circuit,
            config=CryptoEventConfig(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps),
        )
        print("FlyDeck visual agent - CRYPTO EVENT POLICY")
        print(f"visual neurons: {len(circuit.neurons)}")
        print(f"visual edges: {len(circuit.edges)}")
        for name, result in zip(("TRAIN", "VALIDATION", "TEST"), metrics):
            print(f"\n{name}")
            print(f"rounds: {result.rounds}")
            print(f"signals: {result.signals}")
            print(f"positive/negative/flat: {result.positive_signals}/{result.negative_signals}/{result.flat_signals}")
            print(f"total return net: {result.total_return:.4%}")
            print(f"max drawdown: {result.max_drawdown:.4%}")
            print(f"volatility: {result.volatility:.6f}")
            print(f"sharpe-like: {result.sharpe_like:.4f}")
            print(f"return/max drawdown: {result.economic.return_over_drawdown:.4f}")
            print(f"CVaR 95%/99%: {result.economic.cvar_95:.6f}/{result.economic.cvar_99:.6f}")
            print(f"profit factor: {result.economic.profit_factor:.4f}")
            print(f"ruin probability: {result.economic.ruin_probability:.4%}")
            print(f"recovery periods: {result.economic.recovery_periods}")
            print(f"turnover: {result.economic.turnover:.4f}")
            print(f"total cost: {result.economic.total_cost:.6f}")
            print(f"hit rate: {result.hit_rate:.3%}")
            print(f"average position: {result.average_position:.4f}")
            print(f"average horizon: {result.average_horizon:.2f} candles")
            print(f"labels UP/DOWN/WAIT: {result.labels_up}/{result.labels_down}/{result.labels_wait}")
            print(f"temporal matches: {result.temporal_matches}")
            print(f"temporal attention: {result.temporal_attention:.4f}")
        return 0
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
        print(f"multiclass brier score (UP/DOWN/WAIT): {metrics.multiclass_brier_score:.6f}")
        print(f"multiclass expected calibration error: {metrics.multiclass_expected_calibration_error:.6f}")
        print(f"regimes: {dict(metrics.regimes)}")
        for regime, regime_rounds, entries, accuracy, brier, ece in metrics.regime_metrics:
            print(f"  {regime}: rounds={regime_rounds} entered={entries} accuracy={accuracy:.3%} brier={brier:.6f} ece={ece:.6f}")
        print(f"wait reasons: {dict(metrics.wait_reasons)}")
        print(f"UP/DOWN/WAIT: {metrics.up}/{metrics.down}/{metrics.wait}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
