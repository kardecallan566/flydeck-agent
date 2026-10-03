"""Strictly prospective, frozen checkpoint evaluation with persistent paper equity.

A single next-period CSV is evaluated once by the same *frozen* 100 candidate
states produced in the development run. Supports a truly paired ablation:
run frozen Fly-trained weights with the signal present AND masked to zero,
without retraining, alongside an independently trained OHLCV control.

Research-only. No orders, signing, browser automation or wallet operations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bnb_prediction_data_runner import BNBPredictionDataset
from .evolution import (
    Candidate, EvolutionSettings, FEATURE_NAMES, _block, _labels, _save_csv,
    market_features,
)
from .pancakeswap_targets import PancakeAlignment


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_checkpoint(path: Path, *, expect_fly: bool) -> tuple[dict, list[Candidate]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("research_only") is not True:
        raise ValueError(f"unsupported or untrusted checkpoint schema: {path}")
    if bool(payload.get("has_shared_fly_signal")) != expect_fly:
        raise ValueError(f"checkpoint Fly feature mode mismatch: {path}")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("feature schema mismatch: retrain on this release")
    if not payload.get("frozen_before_recent"):
        raise ValueError("checkpoint was not frozen before recent evaluation")
    configs = EvolutionSettings(**payload["settings"])
    agents: list[Candidate] = []
    seen: set[str] = set()
    for entry in payload["agents"]:
        agent_id = str(entry["agent_id"])
        weights = np.asarray(entry["weights"], dtype=np.float64)
        mask = np.asarray(entry["mask"], dtype=np.float64)
        if (agent_id in seen or weights.shape != (len(FEATURE_NAMES),)
            or mask.shape != (len(FEATURE_NAMES),)
            or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(mask))
            or np.any(mask < 0) or np.any(mask > 1)):
            raise ValueError("invalid or duplicate candidate in checkpoint")
        seen.add(agent_id)
        agents.append(Candidate(
            agent_id=agent_id, family=str(entry["family"]),
            parent_id=str(entry["parent_id"]), generation=int(entry["generation"]),
            weights=weights, mask=mask, learning_rate=float(entry["learning_rate"]),
            l2=float(entry["l2"]), threshold=float(entry["threshold"]),
        ))
    if len(agents) != configs.population:
        raise ValueError("population size in checkpoint does not match configuration")
    finalists = payload["finalists_selected_on_validation"]
    if not isinstance(finalists, list) or not all(x in seen for x in finalists):
        raise ValueError("invalid preselected finalist IDs")
    return payload, agents


def _summary(rows: list[dict]) -> dict:
    return {
        "agents": len(rows),
        "positive_this_window": sum(r["window_return_pct"] > 0 for r in rows),
        "above_initial_capital": sum(r["cumulative_return_pct"] > 0 for r in rows),
        "median_window_return_pct": statistics.median(r["window_return_pct"] for r in rows),
        "median_accuracy": statistics.median(r["accuracy"] for r in rows),
        "median_accuracy_active_only": (
            statistics.median(r["accuracy"] for r in rows if r["entered"] > 0)
            if any(r["entered"] > 0 for r in rows) else None
        ),
        "agents_without_entries": sum(r["entered"] == 0 for r in rows),
        "median_coverage": statistics.median(r["coverage"] for r in rows),
        "median_peak_to_trough_drawdown": statistics.median(r["max_drawdown"] for r in rows),
        "wilson_lower_above_scenario_break_even_count": sum(
            r["statistical_evidence"] for r in rows
        ),
    }


def directional_baselines(
    data: BNBPredictionDataset, y: np.ndarray, ready: np.ndarray,
    settings: EvolutionSettings, *, variable_odds: bool,
) -> dict:
    """Honest unselective baselines on the SAME resolved timestamps.

    Fixed payout returns are ONLY hypothetical; pre-lock variable payout
    cannot be reconstructed from direction alone.
    """
    indices = [
        i for i in range(32, data.size)
        if y[i] >= 0 and ready[i] < data.size
    ]
    n = len(indices)
    up = sum(int(y[i] == 1) for i in indices)
    down = n - up
    be = (1 + settings.gas_fraction_of_stake) / (
        settings.scenario_gross_odds * (1 - settings.scenario_fee)
    )
    def policy(wins: int) -> dict:
        return {
            "entries": n, "accuracy": wins / n if n else None,
            "hypothetical_return_pct": (
                None if variable_odds or not n
                else 100 * (
                    (1 + settings.stake_fraction * (
                        settings.scenario_gross_odds * (1 - settings.scenario_fee)
                        - 1 - settings.gas_fraction_of_stake
                    )) ** wins
                    * (1 - settings.stake_fraction * (
                        1 + settings.gas_fraction_of_stake
                    )) ** (n - wins)
                    - 1
                )
            ),
        }
    return {
        "same_resolved_observations": n,
        "always_up": policy(up),
        "always_down": policy(down),
        "always_wait_return_pct": 0.0,
        "up_base_rate": up / n if n else None,
        "fixed_payout_break_even_probability": be if not variable_odds else None,
        "payout_note": (
            "No economic baseline without per-decision actual final pool payouts."
            if variable_odds else
            "Hypothetical fixed 2x/fee/gas settings; NOT realizable PancakeSwap PnL."
        ),
    }


def frozen_forward_arm(
    *,
    checkpoint_file: Path, data: BNBPredictionDataset,
    output: Path, features: np.ndarray, after_timestamp_ms: int,
    source_csv_sha256: str, name: str, expect_fly: bool,
    alignment: PancakeAlignment | None = None,
    odds_by_epoch: dict[int, tuple[float, float]] | None = None,
    resume_state: Path | None = None,
    trace_all: bool = False,
    cumulative_inspected: bool = False,
) -> dict:
    """Fails closed on previously seen timestamps and non-frozen model state."""
    checkpoint, agents = load_checkpoint(checkpoint_file, expect_fly=expect_fly)
    cp_hash = sha256_file(checkpoint_file)
    if data.size < 35:
        raise ValueError("forward test requires at least 35 closed candles")
    if features.shape != (data.size, len(FEATURE_NAMES)):
        raise ValueError("feature matrix has the wrong dimensions")
    if not np.all(np.isfinite(features)):
        raise ValueError("forward features must all be finite")
    if cumulative_inspected and resume_state is not None:
        raise ValueError("cumulative replay already includes past candles: omit --resume-root")
    history_end = int(checkpoint["last_historical_candle_open_ms"])
    if data.timestamps[0] <= max(history_end, after_timestamp_ms):
        raise ValueError(
            f"fresh data overlaps training/previously inspected candles: "
            f"first={data.timestamps[0]} last_known={max(history_end, after_timestamp_ms)}"
        )
    previous: dict[str, dict] = {}
    if resume_state is not None:
        saved = json.loads(resume_state.read_text(encoding="utf-8"))
        if (saved.get("checkpoint_sha256") != cp_hash
            or saved.get("arm") != name
            or saved.get("source_target") != data.target_name):
            raise ValueError("resume state does not match frozen model/arm/target")
        if data.timestamps[0] <= int(saved["last_test_candle_open_ms"]):
            raise ValueError("resume dataset overlaps previously tested time window")
        previous = saved["agents"]

    # No training anywhere in forward: even WAIT outcomes are scored only.
    x = features
    y, ready, epochs = _labels(data, alignment)
    config = EvolutionSettings(**checkpoint["settings"])
    baseline = directional_baselines(
        data, y, ready, config, variable_odds=odds_by_epoch is not None,
    )
    trace: list[dict] = []
    state_out: dict[str, dict] = {}
    results = _block(
        agents, x, y, ready, 32, data.size, stage="new_forward",
        block=0, settings=config,
        learn=False, odds=odds_by_epoch, epochs=epochs,
        trace_rows=trace,
        trace_agents=(
            {a.agent_id for a in agents}
            if trace_all else set(checkpoint["finalists_selected_on_validation"])
        ),
        timestamp_ms=data.timestamps,
        initial_state=previous or None, state_out=state_out,
    )
    by_id = {a.agent_id: a for a in agents}
    table: list[dict] = []
    for metric in results:
        old_eq = float(previous.get(metric.agent_id, {}).get("equity", 100.0))
        eq = state_out[metric.agent_id]["equity"]
        row = asdict(metric)
        row.update({
            "model_checkpoint_sha256": cp_hash,
            "preselected_finalist": metric.agent_id in checkpoint["finalists_selected_on_validation"],
            "window_return_pct": 100 * (eq / old_eq - 1),
            "cumulative_return_pct": 100 * (eq / 100 - 1),
            "window_equity_start": old_eq,
            "window_equity_end": eq,
            "training_generation": by_id[metric.agent_id].generation,
        })
        table.append(row)

    if output.exists() and (output / "forward_summary.json").exists():
        raise FileExistsError(f"this forward run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _save_csv(output / "all_agents_forward.csv", table)
    _save_csv(output / "preselected_decisions.csv", trace)
    preselected = [r for r in table if r["preselected_finalist"]]
    summary = {
        "research_only": True,
        "arm": name,
        "checkpoint_sha256": cp_hash,
        "source_target": data.target_name,
        "cross_target_test": checkpoint["source_target"] != data.target_name,
        "economic_mode": (
            "prelock_snapshot" if odds_by_epoch is not None
            else "illustrative_scenario_not_actual_pnl"
        ),
        "new_candles": data.size,
        "cumulative_already_inspected": cumulative_inspected,
        "independent_new_holdout": not cumulative_inspected,
        "replayed_from_initial_frozen_checkpoint": cumulative_inspected,
        "skipped_causal_warmup_candles": 32,
        "actually_evaluable_observations": baseline["same_resolved_observations"],
        "risk_threshold_feasible_this_window": baseline["same_resolved_observations"] >= config.min_entries,
        "directional_baselines": baseline,
        "first_new_open_ms": int(data.timestamps[0]),
        "first_evaluated_open_ms": int(data.timestamps[32]),
        "last_downloaded_open_ms": int(data.timestamps[-1]),
        "previously_inspected_last_open_ms": after_timestamp_ms,
        "source_csv_sha256": source_csv_sha256,
        "evaluation_without_weight_updates": True,
        "selected_before_forward": checkpoint["finalists_selected_on_validation"],
        "full_cohort": _summary(table),
        "preselected_finalists": _summary(preselected) if preselected else None,
        "model_search_correction": "not_established",
        "live_trading_approved": False,
    }
    (output / "forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    next_state = {
        "schema_version": 1,
        "cumulative_replay_source": cumulative_inspected,
        "arm": name, "checkpoint_sha256": cp_hash,
        "source_target": data.target_name,
        "last_test_candle_open_ms": int(data.timestamps[-1]),
        "agents": state_out,
    }
    (output / "forward_state.json").write_text(
        json.dumps(next_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize_frozen_comparison(
    treatment: dict, control: dict, masked_signal: dict | None, output: Path,
) -> dict:
    if treatment["source_csv_sha256"] != control["source_csv_sha256"]:
        raise ValueError("control and treatment evaluated different candle files")
    keys = ("positive_this_window", "median_window_return_pct",
            "median_accuracy", "median_coverage")
    t, c = treatment["full_cohort"], control["full_cohort"]
    report = {
        "research_only": True,
        "same_new_candles_and_frozen_parameters": True,
        "new_data_sha256": treatment["source_csv_sha256"],
        "with_fly": treatment,
        "without_fly": control,
        "with_minus_without": {key: t[key] - c[key] for key in keys},
        "note": (
            "Differences across evolved populations do not prove causal contribution. "
            "The optional masked-signal arm uses the exact same Fly-trained agents, "
            "with the Fly input replaced by zero; an out-of-distribution diagnostic."
        ),
        "does_not_establish_economic_edge": True,
    }
    if masked_signal is not None:
        if treatment["checkpoint_sha256"] != masked_signal["checkpoint_sha256"]:
            raise ValueError("masked-signal checkpoint must match Fly checkpoint")
        mt = masked_signal["full_cohort"]
        report["fly_trained_signal_masked"] = masked_signal
        report["signal_present_minus_masked_on_same_agents"] = {
            key: t[key] - mt[key] for key in keys
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
