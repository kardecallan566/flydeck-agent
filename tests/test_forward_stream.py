"""Research regressions: continuous candle provenance, paired diagnostics and honest baselines.

Fully offline; synthetic OHLCV only, no MaleCNS circuit, network or wallet.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.data.market_cache import read_csv, write_csv
from flydeck.data.market_data import MarketCandle, dataset_from_candles
from flydeck.evolution import EvolutionSettings, market_features, run_evolution
from flydeck.forward_append_cli import append_closed_candles
from flydeck.forward_diagnostics_cli import diagnose
from flydeck.forward_eval import frozen_forward_arm, summarize_frozen_comparison


def sample(n: int, start=1_000_000) -> BNBPredictionDataset:
    t = tuple(start + 300_000 * i for i in range(n))
    close = tuple(100 + .008 * i + .3 * float(np.sin(i / 3)) for i in range(n))
    return BNBPredictionDataset(
        timestamps=t, opens=close,
        highs=tuple(c + .2 for c in close),
        lows=tuple(c - .2 for c in close),
        closes=close, volumes=tuple(8.0 + i % 12 for i in range(n)),
    )


def to_csv(data: BNBPredictionDataset, path: Path) -> None:
    candles = tuple(
        MarketCandle(
            timestamp=data.timestamps[i], open=data.opens[i],
            high=data.highs[i], low=data.lows[i],
            close=data.closes[i], volume=data.volumes[i],
        ) for i in range(data.size)
    )
    dataset = dataset_from_candles(
        candles, symbol="BNBUSDT", interval="5m", source="test",
        interval_ms=300_000,
    )
    write_csv(dataset, path)


def frozen_pair(tmp_path: Path) -> tuple[BNBPredictionDataset, Path, Path]:
    history = sample(360)
    settings = EvolutionSettings(
        population=100, block_size=40, min_entries=1,
    )
    root = tmp_path / "training"
    run_evolution(
        history, settings, output=root / "fly",
        fly_signal=np.linspace(-.1, .2, history.size, dtype=np.float32),
    )
    run_evolution(history, settings, output=root / "control")
    return (
        history, root / "fly" / "population_checkpoint.json",
        root / "control" / "population_checkpoint.json",
    )


def test_append_preserves_all_candles_and_manifest_and_rejects_replays(tmp_path):
    old, fresh = tmp_path / "smoke.csv", tmp_path / "batch02.csv"
    to_csv(sample(95), old)
    to_csv(sample(40, start=1_000_000 + 95 * 300_000), fresh)
    result = append_closed_candles(old, fresh, tmp_path / "full.csv")
    assert result["total_rows"] == 135
    assert result["expected_labelled_decisions"] == 102
    assert result["causal_warmup_once"] == 32
    assert result["cumulative_already_inspected"]
    assert (tmp_path / "full.manifest.json").is_file()
    combined = read_csv(tmp_path / "full.csv", symbol="BNBUSDT", interval="5m")
    assert len(combined.candles) == 135
    assert combined.candles[95].timestamp == combined.candles[94].timestamp + 300_000
    with pytest.raises(FileExistsError, match="refusing"):
        append_closed_candles(old, fresh, tmp_path / "full.csv")
    with pytest.raises(ValueError, match="overlap or gap"):
        append_closed_candles(old, old, tmp_path / "reject_overlap.csv")
    to_csv(sample(40, start=1_000_000 + 96 * 300_000), tmp_path / "gap.csv")
    with pytest.raises(ValueError, match="overlap or gap"):
        append_closed_candles(old, tmp_path / "gap.csv", tmp_path / "reject_gap.csv")
    assert not (tmp_path / "reject_gap.csv").exists()


def test_95_candle_result_is_62_resolved_not_61_and_has_honest_baselines(tmp_path):
    history, fly_cp, _ = frozen_pair(tmp_path)
    recent = sample(95, start=history.timestamps[-1] + 300_000)
    result = frozen_forward_arm(
        checkpoint_file=fly_cp, data=recent,
        features=market_features(recent, np.zeros(recent.size, dtype=np.float32)),
        output=tmp_path / "forward95", after_timestamp_ms=history.timestamps[-1],
        source_csv_sha256="simulated95", name="with_fly", expect_fly=True,
    )
    assert result["actually_evaluable_observations"] == 62
    assert result["directional_baselines"]["same_resolved_observations"] == 62
    assert (result["directional_baselines"]["always_up"]["entries"] ==
            result["directional_baselines"]["always_down"]["entries"] == 62)
    assert result["directional_baselines"]["always_wait_return_pct"] == 0.0
    assert result["independent_new_holdout"] is True
    assert all(
        row["statistical_evidence"] == "False"
        for row in csv.DictReader((tmp_path / "forward95" / "all_agents_forward.csv").open())
        if int(row["entered"]) < 1
    )


def test_three_arm_paired_diagnostic_no_entry_agents_excluded(tmp_path):
    history, fly_cp, plain_cp = frozen_pair(tmp_path)
    recent = sample(95, start=history.timestamps[-1] + 300_000)
    results = {}
    for name, cp, fly in (
        ("with_fly", fly_cp, np.ones(recent.size, dtype=np.float32) * .25),
        ("without_fly", plain_cp, None),
        ("with_fly_signal_masked", fly_cp, None),
    ):
        results[name] = frozen_forward_arm(
            checkpoint_file=cp, data=recent,
            features=market_features(recent, fly),
            output=tmp_path / "run" / name,
            after_timestamp_ms=history.timestamps[-1],
            source_csv_sha256="the-same-95",
            name=name, expect_fly=name != "without_fly",
        )
    summarize_frozen_comparison(
        results["with_fly"], results["without_fly"],
        results["with_fly_signal_masked"], tmp_path / "run",
    )
    audit = diagnose(tmp_path / "run", output=tmp_path / "audit.json")
    assert audit["evaluated_opportunities"] == 62
    assert audit["risk_threshold_possible_this_window"] is True  # synthetic fixture sets min_entries=1
    assert audit["research_80_entry_minimum_possible"] is False
    assert audit["same_weight_neural_input_diagnostic"]["same_frozen_checkpoint_and_candles"]
    paired = audit["same_weight_neural_input_diagnostic"]["original_finalists_only"]
    assert paired["paired_decision_opportunities_correlated_across_agents"] == 5 * 62
    assert (paired["changed_direction_when_both_entered"] +
            paired["entry_vs_wait_disagreements"] == paired["changed_actions"])
    assert 0 <= paired["mean_brier_signal_on"] <= 1
    assert 0 <= paired["mean_brier_signal_masked"] <= 1
    for name in ("with_fly", "without_fly"):
        cal = audit[name]["preselected_probability_calibration"]
        assert cal["agent_time_observations_correlated"] == 5 * 62
        assert cal["includes_wait_predictions"]
        arm = audit[name]["all"]
        assert arm["agents_without_entries"] + arm["agents_with_entries"] == 100
        assert arm["median_accuracy_among_active"] is None or 0 <= arm["median_accuracy_among_active"] <= 1
    with pytest.raises(FileExistsError):
        diagnose(tmp_path / "run", output=tmp_path / "audit.json")


def test_cumulative_replay_once_preserves_all_possible_labels(tmp_path):
    history, fly_cp, _ = frozen_pair(tmp_path)
    start = history.timestamps[-1] + 300_000
    prior = sample(95, start)
    new = sample(40, start + 95 * 300_000)
    whole = BNBPredictionDataset(
        timestamps=prior.timestamps + new.timestamps,
        opens=prior.opens + new.opens,
        highs=prior.highs + new.highs,
        lows=prior.lows + new.lows,
        closes=prior.closes + new.closes,
        volumes=prior.volumes + new.volumes,
    )
    single = frozen_forward_arm(
        checkpoint_file=fly_cp, data=whole,
        features=market_features(whole, np.zeros(whole.size, dtype=np.float32)),
        output=tmp_path / "cumulative", after_timestamp_ms=history.timestamps[-1],
        source_csv_sha256="cumulative", name="with_fly", expect_fly=True,
        cumulative_inspected=True,
    )
    assert single["cumulative_already_inspected"]
    assert not single["independent_new_holdout"]
    assert single["actually_evaluable_observations"] == 102
    with pytest.raises(ValueError, match="omit --resume-root"):
        frozen_forward_arm(
            checkpoint_file=fly_cp, data=whole,
            features=market_features(whole),
            output=tmp_path / "reject", after_timestamp_ms=history.timestamps[-1],
            source_csv_sha256="cumulative", name="with_fly", expect_fly=True,
            cumulative_inspected=True,
            resume_state=tmp_path / "some_previous_state.json",
        )
