"""Reproducible v3/v4 diagnosis of directional drift and frozen MaleCNS utility.

Historical development labels determine a descriptive PRE-audit base-rate
reference; neither already-inspected audit nor forward outcomes fit a model.
Research only: no candidate promotion, training or real transactions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .bnb_prediction_data_runner import load_bnb_5m_csv
from .evolution import FEATURE_NAMES, FLY_CACHE_VERSION, _labels
from .forward_diagnostics_cli import arm_audit, paired_same_weights
from .forward_eval import load_checkpoint, sha256_file


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _blocks(data, settings) -> tuple[list[dict], float]:
    total_blocks = (
        settings.warmup_blocks + settings.development_blocks
        + settings.validation_blocks + settings.audit_blocks
    )
    labels, ready, _ = _labels(data, alignment=None)
    windows = []
    developed_positive = developed_total = 0
    for block in range(total_blocks):
        first = 31 if block == 0 else block * settings.block_size
        end = min(data.size - 1, (block + 1) * settings.block_size)
        if block < settings.warmup_blocks:
            phase = "train"
        elif block < settings.warmup_blocks + settings.development_blocks:
            phase = "development"
        elif block < total_blocks - settings.audit_blocks:
            phase = "validation"
        else:
            phase = "historical_audit"
        positions = [
            k for k in range(first, end)
            if labels[k] >= 0 and ready[k] < end
        ]
        up = sum(int(labels[k] == 1) for k in positions)
        count = len(positions)
        row = {
            "block": block, "phase": phase,
            "first_open_ms": int(data.timestamps[first]),
            "last_open_ms": int(data.timestamps[end - 1]),
            "evaluable": count, "up": up, "down": count - up,
            "up_rate": _ratio(up, count),
            "ties_or_unresolved": end - first - count,
            "already_inspected": phase in {"validation", "historical_audit"},
        }
        windows.append(row)
        if phase in ("train", "development"):
            developed_positive += up
            developed_total += count
    if not developed_total:
        raise ValueError("no development-only settled outcomes")
    return windows, developed_positive / developed_total


def _signal_audit(
    *, cache_file: Path, history_file: Path, checkpoint: dict,
    stamps: tuple[int, ...], closes: tuple[float, ...], agents,
) -> dict:
    with np.load(cache_file, allow_pickle=False) as cache:
        signal = np.asarray(cache["signal"], dtype=np.float64)
        times = np.asarray(cache["timestamps"], dtype=np.int64)
        fingerprint = str(cache["fingerprint"].item())
    required = ":".join((
        FLY_CACHE_VERSION,
        sha256_file(history_file),
        str(checkpoint["circuit_sha256"]),
        "32",
    ))
    if fingerprint != required:
        raise ValueError(
            "MaleCNS historical signal cache fingerprint does not match "
            "the frozen v3/v4 history, circuit and 32-candle context"
        )
    if signal.shape != (len(stamps),) or not np.array_equal(
        times, np.asarray(stamps, dtype=np.int64)
    ):
        raise ValueError("MaleCNS signal and history timestamps mismatch")
    if not np.all(np.isfinite(signal)):
        raise ValueError("non-finite MaleCNS cache signal")
    eligible = signal[32:-1]
    targets = np.sign(
        np.asarray(closes[33:], dtype=np.float64)
        - np.asarray(closes[32:-1], dtype=np.float64)
    )
    good = targets != 0
    both = eligible[good]
    target = (targets[good] > 0).astype(float)
    corr = (
        float(np.corrcoef(both, target)[0, 1])
        if len(both) > 3 and np.std(both) > 1e-9
        and np.std(target) > 1e-9 else None
    )
    by_finalist = []
    finalists = set(checkpoint["finalists_selected_on_validation"])
    for agent in agents:
        if agent.agent_id not in finalists:
            continue
        weight = float(agent.weights[-1] * agent.mask[-1])
        other = np.abs(agent.weights[:-1] * agent.mask[:-1])
        by_finalist.append({
            "agent_id": agent.agent_id,
            "family": agent.family,
            "threshold": agent.threshold,
            "effective_intercept": float(agent.weights[0] * agent.mask[0]),
            "effective_fly_weight": weight,
            "max_observed_abs_logit_shift_from_fly": (
                float(np.max(np.abs(eligible * weight))) if len(eligible) else 0.0
            ),
            "median_abs_non_fly_effective_weight": float(np.median(other)),
            "fly_logit_ratio_to_non_fly_weight_median": (
                abs(weight) / float(np.median(other))
                if float(np.median(other)) > 1e-12 else None
            ),
        })
    return {
        "cache_fingerprint_verified": True,
        "non_warmup_samples": len(eligible),
        "signal_zero_fraction": float(np.mean(eligible == 0)) if len(eligible) else None,
        "signal_mean": float(np.mean(eligible)) if len(eligible) else None,
        "signal_std": float(np.std(eligible)) if len(eligible) else None,
        "signal_p05_p50_p95": [float(v) for v in np.percentile(eligible, [5, 50, 95])],
        "same_observation_signal_vs_next_closed_candle_up_correlation": corr,
        "correlation_only_descriptive_and_in_sample": True,
        "finalist_effective_feature_weights": by_finalist,
    }


def investigate(
    *, history: Path, with_checkpoint: Path, without_checkpoint: Path,
    forward_root: Path, fly_cache: Path | None = None,
    output: Path | None = None,
) -> dict:
    history = Path(history)
    cp_fly, fly_agents = load_checkpoint(Path(with_checkpoint), expect_fly=True)
    cp_no, no_agents = load_checkpoint(Path(without_checkpoint), expect_fly=False)
    digest = sha256_file(history)
    for checkpoint in (cp_fly, cp_no):
        if checkpoint.get("source_candles_sha256") != digest:
            raise ValueError(
                "checkpoint and historical CSV SHA-256 mismatch: no mix of histories"
            )
        if checkpoint.get("source_target") != "binance-close-t+1":
            raise ValueError("this investigation currently requires binance-close-t+1")
    if (cp_fly["settings"] != cp_no["settings"]
        or cp_fly["last_historical_candle_open_ms"]
           != cp_no["last_historical_candle_open_ms"]):
        raise ValueError("Fly and OHLCV checkpoints must have matched setup")
    data = load_bnb_5m_csv(history)
    if data.timestamps[-1] != cp_fly["last_historical_candle_open_ms"]:
        raise ValueError("history timestamp does not match frozen checkpoint")

    from .evolution import EvolutionSettings
    settings = EvolutionSettings(**cp_fly["settings"])
    windows, development_up = _blocks(data, settings)
    fly_arm = arm_audit(forward_root / "with_fly")
    no_arm = arm_audit(forward_root / "without_fly")
    if (fly_arm["checkpoint_sha256"] != sha256_file(with_checkpoint)
        or no_arm["checkpoint_sha256"] != sha256_file(without_checkpoint)):
        raise ValueError("forward results do not match supplied frozen checkpoints")
    if fly_arm["source_sha256"] != no_arm["source_sha256"]:
        raise ValueError("forward arms are from different market datasets")
    forward = json.loads(
        (forward_root / "with_fly" / "forward_summary.json").read_text(encoding="utf-8")
    )
    base = forward["directional_baselines"]
    forward_up = base["up_base_rate"]
    if forward_up is None:
        raise ValueError("forward evaluation contains no labelled observations")
    masked = forward_root / "with_fly_signal_masked"
    paired = paired_same_weights(forward_root / "with_fly", masked) if masked.is_dir() else None
    def intercept_stats(agents):
        xs = [float(a.weights[0] * a.mask[0]) for a in agents]
        return {
            "agents": len(xs),
            "negative_intercepts": sum(v < 0 for v in xs),
            "median_effective_intercept": statistics.median(xs),
            "preselected": [
                {
                    "agent_id": a.agent_id, "family": a.family,
                    "effective_intercept": float(a.weights[0] * a.mask[0]),
                    "neutral_features_probability_up": float(
                        1 / (1 + math.exp(-float(a.weights[0] * a.mask[0])))
                    ),
                }
                for a in agents if a.agent_id in (
                    cp_fly["finalists_selected_on_validation"] if agents is fly_agents
                    else cp_no["finalists_selected_on_validation"]
                )
            ],
        }
    baseline_brier = (
        forward_up * (1 - development_up) ** 2
        + (1 - forward_up) * development_up ** 2
    )
    result = {
        "research_only": True,
        "checkpoint_selection_not_modified": True,
        "history_sha256": digest,
        "forward_sha256": fly_arm["source_sha256"],
        "forward_cumulative_already_inspected": fly_arm["cumulative_already_inspected"],
        "historical_directional_regimes": windows,
        "development_only_up_prior": development_up,
        "forward_up_rate": forward_up,
        "forward_up_rate_minus_development_up_prior": forward_up - development_up,
        "forward_only_retroactive_reference": base,
        "forecast_brier_constant_prior_frozen_from_development": baseline_brier,
        "forecast_brier_constant_half": 0.25,
        "fly_checkpoint_intercepts": intercept_stats(fly_agents),
        "ohlcv_checkpoint_intercepts": intercept_stats(no_agents),
        "forward_with_fly_preselected": {
            "group": fly_arm["preselected_validation_finalists"],
            "rows": fly_arm["preselected_rows"],
            "calibration": fly_arm["preselected_probability_calibration"],
        },
        "forward_without_fly_preselected": {
            "group": no_arm["preselected_validation_finalists"],
            "rows": no_arm["preselected_rows"],
            "calibration": no_arm["preselected_probability_calibration"],
        },
        "paired_same_frozen_weights": (
            {
                "agents": paired["agents"],
                "changed_entry_counts": paired["agents_with_changed_entry_counts"],
                "median_return_delta_pp": paired["median_agent_return_delta_percentage_points"],
                "original_finalists": paired["original_finalists_only"],
            } if paired is not None else None
        ),
        "selection_warning": (
            "Historical audit and the cumulative 300-candle stream have already "
            "been seen. Do not select/promote a strategy using these observations."
        ),
    }
    if fly_cache is not None:
        result["historical_neural_signal"] = _signal_audit(
            cache_file=Path(fly_cache), history_file=history,
            checkpoint=cp_fly, stamps=data.timestamps, closes=data.closes,
            agents=fly_agents,
        )
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite prior diagnosis: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", dir=output.parent, encoding="utf-8", delete=False,
            suffix=".tmp",
        ) as tmp:
            candidate = Path(tmp.name)
            json.dump(result, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
        os.replace(candidate, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline causal-window, DOWN-bias and MaleCNS audit; NEVER trains or promotes agents."
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--with-checkpoint", type=Path, required=True)
    parser.add_argument("--without-checkpoint", type=Path, required=True)
    parser.add_argument("--forward-run", type=Path, required=True)
    parser.add_argument("--fly-cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = investigate(
            history=args.history, with_checkpoint=args.with_checkpoint,
            without_checkpoint=args.without_checkpoint,
            forward_root=args.forward_run, fly_cache=args.fly_cache,
            output=args.output,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Investigation aborted: {exc}\n")
    print("Historical development-only UP prior:",
          f"{report['development_only_up_prior']:.2%}")
    print("Already-inspected forward UP frequency:",
          f"{report['forward_up_rate']:.2%}")
    for name in ("fly_checkpoint_intercepts", "ohlcv_checkpoint_intercepts"):
        p = report[name]
        print(f"{name}: negative intercepts={p['negative_intercepts']}/"
              f"{p['agents']}, median={p['median_effective_intercept']:.4f}")
    if "historical_neural_signal" in report:
        signal = report["historical_neural_signal"]
        print("Frozen MaleCNS historical signal STD:",
              f"{signal['signal_std']:.6f}")
    if args.output:
        print("Auditable investigation saved:", args.output)
    print("Already-inspected data is descriptive ONLY; no promotion or real trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
