from __future__ import annotations

import argparse
from pathlib import Path

from .ablation_suite import format_ablation_table, run_ablation_suite, run_agent_over_range
from .bnb_prediction_data_runner import download_bnb_5m_csv, load_bnb_5m_csv
from .param_optimizer import format_optimization_results, run_grid_search
from .receptive_fields import infer_receptive_fields
from .scientific_benchmarks import (
    AlwaysDownBaseline,
    AlwaysUpBaseline,
    ConfusionMatrix,
    MomentumBaseline,
    PreviousDirectionBaseline,
    RandomBaseline,
    evaluate_baseline,
    evaluate_by_regimes,
    evaluate_predictions,
)
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


def format_matrix_row(m: ConfusionMatrix) -> str:
    edge = m.pancakeswap_expectancy()
    return (
        f"{m.name:<32} | "
        f"{m.entered_rounds:>5}/{m.total_rounds:<5} | "
        f"{m.coverage:>6.1%} | "
        f"{m.accuracy:>6.1%} | "
        f"{m.precision_up:>6.1%} | "
        f"{m.precision_down:>6.1%} | "
        f"{m.f1_score:>6.3f} | "
        f"{edge:>+7.2%}"
    )


def print_comparison_table(title: str, matrices: tuple[ConfusionMatrix, ...]) -> None:
    print(f"\n=== {title} ===")
    header = f"{'Model / Baseline':<32} | {'Entered/Tot':<11} | {'Cover':>6} | {'Acc':>6} | {'P(UP)':>6} | {'P(DN)':>6} | {'F1':>6} | {'PCS Edge':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for m in matrices:
        print(format_matrix_row(m))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific validation and ablation suite for FlyDeck Agent")
    parser.add_argument("--circuit", required=True, type=Path, help="Path to MaleCNS visual circuit JSON")
    parser.add_argument("--data", type=Path, help="Path to BNB 5m CSV")
    parser.add_argument("--download", type=Path, help="Download BNB 5m dataset from Binance Vision to path")
    parser.add_argument("--limit", default=3000, type=int, help="Number of candles to download")
    parser.add_argument("--confidence", default=0.15, type=float, help="Decision confidence threshold")
    parser.add_argument("--context", default=32, type=int, help="Retina window in candles")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip systematic ablation battery")
    parser.add_argument("--optimize", action="store_true", help="Run grid search optimization on validation split")
    args = parser.parse_args()

    data_path = args.data
    if args.download:
        print(f"Downloading {args.limit} BNB 5m candles to {args.download}...")
        download_bnb_5m_csv(args.download, limit=args.limit)
        data_path = args.download

    if not data_path or not data_path.exists():
        raise SystemExit("Dataset file not found. Provide --data or --download.")

    dataset = load_bnb_5m_csv(data_path)
    usable = dataset.size - 1
    if usable < args.context + 20:
        raise SystemExit(f"Dataset has only {dataset.size} candles; requires at least {args.context + 20}.")

    train_end = int(usable * 0.70)
    val_end = train_end + int(usable * 0.15)
    test_start = val_end
    test_end = usable

    print(f"Loaded BNB 5m dataset: {dataset.size} candles")
    print(f"  Train split:      index {args.context - 1} .. {train_end} ({train_end - args.context + 1} rounds)")
    print(f"  Validation split: index {train_end} .. {val_end} ({val_end - train_end} rounds)")
    print(f"  Test split (OOS): index {test_start} .. {test_end} ({test_end - test_start} rounds)")

    print(f"\nLoading MaleCNS visual circuit: {args.circuit}...")
    circuit = VisualCircuit.load(args.circuit)
    print(f"  Neurons: {len(circuit.neurons)}, Edges: {len(circuit.edges)}")

    print("Inferring retinotopic receptive fields...")
    fields = infer_receptive_fields(circuit, iterations=6)

    # 1. FlyDeck Agent Run on Test Set
    fly_agent = FlyVisualPredictionAgent(
        circuit,
        retina_width=args.context,
        confidence_threshold=args.confidence,
        receptive_fields=fields,
    )
    fly_test_preds = run_agent_over_range(fly_agent, dataset, test_start, test_end, context=args.context)
    test_outcomes = tuple(dataset.outcome(i) for i in range(test_start, test_end))
    fly_matrix = evaluate_predictions("FlyDeck Agent (MaleCNS)", fly_test_preds, test_outcomes)

    # 2. Baselines Run on Test Set
    baselines = (
        ("Lag-1 Persistence (Prev Dir)", PreviousDirectionBaseline()),
        ("Rolling Momentum (6 candles)", MomentumBaseline(window=6)),
        ("Always UP", AlwaysUpBaseline()),
        ("Always DOWN", AlwaysDownBaseline()),
        ("Random Coin Flip", RandomBaseline(seed=123)),
    )
    baseline_matrices = tuple(
        evaluate_baseline(b, dataset, test_start, test_end, name)
        for name, b in baselines
    )

    all_test_matrices = (fly_matrix,) + baseline_matrices
    print_comparison_table("TEST SET EVALUATION (Out-Of-Sample)", all_test_matrices)

    # 3. Market Regime Breakdown for FlyDeck
    regime_results = evaluate_by_regimes("FlyDeck", fly_test_preds, dataset, test_start, test_end)
    regime_matrices = tuple(regime_results.values())
    print_comparison_table("FLYDECK PERFORMANCE BY MARKET REGIME", regime_matrices)

    # 4. Detailed Confusion Matrix for FlyDeck
    print("\n=== FLYDECK TEST SET CONFUSION MATRIX ===")
    print(f"  True UP:    {fly_matrix.true_up:<5} | False UP:   {fly_matrix.false_up:<5}")
    print(f"  True DOWN:  {fly_matrix.true_down:<5} | False DOWN: {fly_matrix.false_down:<5}")
    print(f"  WAITs:      {fly_matrix.waits:<5} (selective abstention rate: {fly_matrix.waits / fly_matrix.total_rounds:.1%})")
    print(f"  PancakeSwap Net Expectancy per bet: {fly_matrix.pancakeswap_expectancy():+.2%}")

    # 5. Systematic Ablation Suite (run on Validation split to avoid test leakage)
    if not args.skip_ablation:
        print("\n=== RUNNING SYSTEMATIC ABLATION BATTERY (Validation Split) ===")
        ablation_results = run_ablation_suite(
            circuit,
            dataset,
            train_end,
            val_end,
            context=args.context,
            confidence_threshold=args.confidence,
            receptive_fields=fields,
        )
        print(format_ablation_table(ablation_results))

    # 6. Grid Search Optimization (on Validation split)
    if args.optimize:
        print("\n=== GRID SEARCH OPTIMIZATION (Validation Split) ===")
        opt_results = run_grid_search(
            circuit,
            dataset,
            train_end,
            val_end,
            context=args.context,
            receptive_fields=fields,
            top_k=10,
            verbose=True,
        )

        if opt_results:
            print("\n--- Top 10 Parameter Sets by PCS Expectancy ---")
            print(format_optimization_results(opt_results))

            # Re-evaluate best params on TEST set (unbiased OOS measurement)
            best = opt_results[0].params
            print(f"\n=== RE-EVALUATION WITH BEST PARAMS ON TEST SET (OOS) ===")
            print(f"  Gamma={best.mutual_inhibition_gamma:.2f}, Beta={best.trend_memory_beta:.2f}, "
                  f"Conf={best.confidence_threshold:.2f}")
            print(f"  Weights: Dir={best.weight_directional:.2f}, Vel={best.weight_velocity:.2f}, "
                  f"Bal={best.weight_on_off_balance:.2f}, Trd={best.weight_trend:.2f}")

            optimized_agent = FlyVisualPredictionAgent(
                circuit,
                retina_width=args.context,
                confidence_threshold=best.confidence_threshold,
                receptive_fields=fields,
                mutual_inhibition_gamma=best.mutual_inhibition_gamma,
                trend_memory_beta=best.trend_memory_beta,
                weight_directional=best.weight_directional,
                weight_velocity=best.weight_velocity,
                weight_on_off_balance=best.weight_on_off_balance,
                weight_trend=best.weight_trend,
            )
            opt_test_preds = run_agent_over_range(optimized_agent, dataset, test_start, test_end, context=args.context)
            opt_test_matrix = evaluate_predictions("FlyDeck (Optimized)", opt_test_preds, test_outcomes)

            print_comparison_table("OPTIMIZED vs DEFAULT on TEST SET (OOS)", (
                fly_matrix,
                opt_test_matrix,
            ))
            opt_edge = opt_test_matrix.pancakeswap_expectancy()
            print(f"\n  Optimized PCS Edge: {opt_edge:+.2%}")
            if opt_edge > 0:
                print("  [+] POSITIVE EXPECTANCY -- agent is viable for PancakeSwap!")
            else:
                print("  [-] Negative expectancy -- further tuning needed.")
        else:
            print("  No viable parameter sets found (all had coverage < 10%).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
