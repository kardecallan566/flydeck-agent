"""No-network tests of the versioned frozen research and prospective pipeline."""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.evolution import (
    EvolutionSettings, FLY_CACHE_VERSION, fly_feature_cache, market_features,
    run_evolution,
)
from flydeck.forward_eval import (
    frozen_forward_arm, load_checkpoint, sha256_file,
    summarize_frozen_comparison,
)


def sample(n: int, start: int = 1_000_000) -> BNBPredictionDataset:
    stamps = tuple(start + 300000 * i for i in range(n))
    closes = tuple(100.0 + 0.006 * i + 0.31 * float(np.sin(i / 3)) for i in range(n))
    return BNBPredictionDataset(
        timestamps=stamps, opens=closes,
        highs=tuple(c + 0.21 for c in closes),
        lows=tuple(c - 0.21 for c in closes),
        closes=closes,
        volumes=tuple(7.0 + i % 9 for i in range(n)),
    )


def checkpoint_pair(tmp_path: Path):
    history = sample(360)
    settings = EvolutionSettings(population=100, block_size=40, min_entries=1)
    fly = np.linspace(-0.3, 0.3, history.size, dtype=np.float32)
    run_evolution(history, settings, output=tmp_path / "training" / "with_fly", fly_signal=fly)
    run_evolution(history, settings, output=tmp_path / "training" / "without_fly")
    return history, (
        tmp_path / "training" / "with_fly" / "population_checkpoint.json",
        tmp_path / "training" / "without_fly" / "population_checkpoint.json",
    )


def test_full_frozen_cache_propagates_to_submodules_and_is_versioned(monkeypatch, tmp_path):
    calls = []
    class FakeCircuit:
        @staticmethod
        def load(path):
            return object()
    class FakeFly:
        neuron_count = 7
        def __init__(self, circuit, confidence_threshold):
            assert confidence_threshold == 0
        def set_learning(self, flag):
            calls.append(flag)
        def perceive(self, prices, volumes=None):
            return None, types.SimpleNamespace(p_up=0.6, p_down=0.3)
    monkeypatch.setitem(sys.modules, "flydeck.visual_circuit",
                        types.SimpleNamespace(VisualCircuit=FakeCircuit))
    monkeypatch.setitem(sys.modules, "flydeck.visual_agent",
                        types.SimpleNamespace(FlyVisualPredictionAgent=FakeFly))
    old = tmp_path / "historic.csv"
    circuit = tmp_path / "circuit.json"
    old.write_text("version 1", encoding="utf-8")
    circuit.write_text('{"version":1}', encoding="utf-8")
    cache = tmp_path / "shared.npz"
    history = sample(40)
    result = fly_feature_cache(
        history, market_file=old, circuit_file=circuit,
        cache_file=cache, rebuild=True,
    )
    assert calls == [False]
    assert np.count_nonzero(result) == 9
    with np.load(cache, allow_pickle=False) as read:
        assert FLY_CACHE_VERSION in str(read["fingerprint"].item())
    again = fly_feature_cache(
        history, market_file=old, circuit_file=circuit, cache_file=cache,
    )
    np.testing.assert_allclose(again, result)
    assert calls == [False]  # no re-inference on valid cached file
    circuit.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        fly_feature_cache(
            history, market_file=old, circuit_file=circuit, cache_file=cache,
        )


def test_frozen_pre_recent_checkpoint_and_finalists_consistent(tmp_path):
    history = sample(360)
    recent = sample(170, start=history.timestamps[-1] + 300000)
    settings = EvolutionSettings(population=100, block_size=40, min_entries=1)
    folder = tmp_path / "training"
    run_evolution(
        history, settings, output=folder,
        recent=recent, recent_holdout=100,
    )
    cp = json.loads((folder / "population_checkpoint.json").read_text())
    finalists = json.loads((folder / "finalists.json").read_text())
    pre_recent = {a["agent_id"]: a for a in cp["agents"]}
    for finalist in finalists:
        assert finalist["weights"] == pre_recent[finalist["agent_id"]]["weights"]
    assert cp["frozen_before_recent"]
    assert len(cp["agents"]) == 100
    assert (folder / "historical_audit_finalist_decisions.csv").exists()
    assert (folder / "recent_sealed_finalist_decisions.csv").exists()


def test_forward_two_populations_counterfactual_and_persistent_bankroll(tmp_path):
    history, (fly_cp, plain_cp) = checkpoint_pair(tmp_path)
    first = sample(145, start=history.timestamps[-1] + 300000)
    second = sample(155, start=first.timestamps[-1] + 300000)
    hash_a = "new-source-a"
    result = {}
    for arm, checkpoint, sig in (
        ("with_fly", fly_cp, np.full(first.size, 0.1, dtype=np.float32)),
        ("without_fly", plain_cp, None),
        ("with_fly_signal_masked", fly_cp, None),
    ):
        result[arm] = frozen_forward_arm(
            checkpoint_file=checkpoint, data=first,
            features=market_features(first, sig),
            output=tmp_path / "first" / arm,
            after_timestamp_ms=history.timestamps[-1],
            source_csv_sha256=hash_a, name=arm,
            expect_fly=(arm != "without_fly"),
        )
        with (tmp_path / "first" / arm / "all_agents_forward.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == 100
        assert result[arm]["evaluation_without_weight_updates"]
        assert all(row["model_checkpoint_sha256"] == sha256_file(checkpoint)
                   for row in table)

    paired = summarize_frozen_comparison(
        result["with_fly"], result["without_fly"],
        result["with_fly_signal_masked"], tmp_path / "first",
    )
    assert paired["same_new_candles_and_frozen_parameters"]
    assert paired["fly_trained_signal_masked"]["checkpoint_sha256"] == sha256_file(fly_cp)
    with pytest.raises(ValueError, match="overlaps"):
        frozen_forward_arm(
            checkpoint_file=fly_cp, data=first,
            features=market_features(first, np.ones(first.size, dtype=np.float32)),
            output=tmp_path / "reject", after_timestamp_ms=first.timestamps[-1],
            source_csv_sha256=hash_a, name="with_fly", expect_fly=True,
        )
    follow = frozen_forward_arm(
        checkpoint_file=fly_cp, data=second,
        features=market_features(second, np.zeros(second.size, dtype=np.float32)),
        output=tmp_path / "second" / "with_fly",
        after_timestamp_ms=first.timestamps[-1],
        source_csv_sha256="new-source-b", name="with_fly", expect_fly=True,
        resume_state=tmp_path / "first" / "with_fly" / "forward_state.json",
    )
    assert follow["previously_inspected_last_open_ms"] == first.timestamps[-1]
    previous = json.loads(
        (tmp_path / "first" / "with_fly" / "forward_state.json").read_text()
    )
    with (tmp_path / "second" / "with_fly" / "all_agents_forward.csv").open() as f:
        later = list(csv.DictReader(f))
    for row in later:
        old_equity = previous["agents"][row["agent_id"]]["equity"]
        assert float(row["window_equity_start"]) == pytest.approx(old_equity)
        assert float(row["cumulative_return_pct"]) == pytest.approx(
            100 * (float(row["window_equity_end"]) / 100 - 1)
        )


def test_checkpoint_rejects_wrong_arm_and_tampered_shape(tmp_path):
    history, (fly_cp, plain_cp) = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(fly_cp, expect_fly=False)
    payload = json.loads(plain_cp.read_text())
    payload["agents"][0]["weights"] = [0.0]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_checkpoint(broken, expect_fly=False)
