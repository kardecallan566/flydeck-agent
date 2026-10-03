"""Offline tests for v4 optional training controls and drift investigation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.data.market_cache import write_csv
from flydeck.data.market_data import MarketCandle, dataset_from_candles
from flydeck.evolution import (
    Candidate, EvolutionSettings, FEATURE_NAMES, FLY_CACHE_VERSION, _block,
    market_features, run_evolution,
)
from flydeck.forward_eval import frozen_forward_arm, sha256_file
from flydeck.investigate_cli import investigate


def sample(n: int, *, start: int = 1_200_000) -> BNBPredictionDataset:
    t = tuple(start + 300_000 * i for i in range(n))
    price = tuple(100 + .009 * i + .25 * np.sin(i / 3) for i in range(n))
    return BNBPredictionDataset(
        timestamps=t, opens=price,
        highs=tuple(float(p + .2) for p in price),
        lows=tuple(float(p - .2) for p in price),
        closes=price, volumes=tuple(5.0 + i % 11 for i in range(n)),
    )


def save(data: BNBPredictionDataset, path: Path):
    rows = tuple(MarketCandle(
        timestamp=data.timestamps[k], open=data.opens[k],
        high=data.highs[k], low=data.lows[k],
        close=data.closes[k], volume=data.volumes[k],
    ) for k in range(data.size))
    write_csv(dataset_from_candles(
        rows, symbol="BNBUSDT", interval="5m", source="synthetic",
        interval_ms=300_000,
    ), path)


def settings(**kwargs):
    return EvolutionSettings(population=100, block_size=40, min_entries=1, **kwargs)


def one_agent():
    return Candidate(
        agent_id="TEST001", family="mixed", parent_id="",
        generation=0, weights=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
        mask=np.ones(len(FEATURE_NAMES)), learning_rate=.1, l2=.01,
        threshold=.1,
    )


def train(ypattern, setting, *, late=None):
    n = len(ypattern)
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    x[:, 0] = 1.0
    y = np.array(ypattern, dtype=np.int8)
    ready = np.arange(n, dtype=np.int64) + 1
    ready[-1] = n + 1
    if late is not None:
        ready[late] = n + 1
    agent = one_agent()
    _block([agent], x, y, ready, 2, n, stage="train",
           block=0, settings=setting, learn=True,
           odds=None, epochs={})
    return agent.weights.copy()


def test_v4_balancing_uses_only_already_settled_classes():
    y = [-1, -1] + [1] * 16 + [0] * 6 + [-1]
    original = train(y, settings())
    explicitly_original = train(y, settings(
        class_balance_alpha=0.0, class_weight_cap=1.0,
        bias_l2_multiplier=1.0,
    ))
    np.testing.assert_array_equal(original, explicitly_original)
    balanced = train(y, settings(
        class_balance_alpha=0.75, class_weight_cap=1.5,
        bias_l2_multiplier=4.0,
    ))
    assert not np.array_equal(original, balanced)
    future_modified = y.copy()
    future_modified[2] = 0
    a = train(y, settings(
        class_balance_alpha=0.75, class_weight_cap=1.5,
        bias_l2_multiplier=4.0,
    ), late=2)
    b = train(future_modified, settings(
        class_balance_alpha=0.75, class_weight_cap=1.5,
        bias_l2_multiplier=4.0,
    ), late=2)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("options,word", [
    ({"class_balance_alpha": -0.1}, "class_balance_alpha"),
    ({"class_balance_alpha": 1.1}, "class_balance_alpha"),
    ({"class_weight_cap": 0.99}, "class_weight_cap"),
    ({"class_weight_cap": 3.1}, "class_weight_cap"),
    ({"bias_l2_multiplier": 0.9}, "bias_l2_multiplier"),
    ({"bias_l2_multiplier": 20.1}, "bias_l2_multiplier"),
])
def test_v4_invalid_hyperparameters_rejected(options, word):
    with pytest.raises(ValueError, match=word):
        settings(**options)


def test_v4_frozen_checkpoints_match_settings_and_offline_investigation(tmp_path):
    history = sample(360)
    history_file = tmp_path / "history.csv"
    save(history, history_file)
    source_hash = sha256_file(history_file)
    setting = settings(
        class_balance_alpha=0.75, class_weight_cap=1.5,
        bias_l2_multiplier=4.0,
    )
    shared = np.sin(np.arange(history.size) / 13).astype(np.float32) * .25
    with_root = tmp_path / "training" / "with_fly"
    no_root = tmp_path / "training" / "without_fly"
    circuit_bytes = b"synthetic-research-only-circuit"
    circuit_file = tmp_path / "circuit.txt"
    circuit_file.write_bytes(circuit_bytes)
    circuit_sha = sha256_file(circuit_file)
    run_evolution(
        history, setting, output=with_root, fly_signal=shared,
        source_candles_sha256=source_hash, circuit_sha256=circuit_sha,
    )
    run_evolution(
        history, setting, output=no_root,
        source_candles_sha256=source_hash,
    )
    fly_cp = with_root / "population_checkpoint.json"
    plain_cp = no_root / "population_checkpoint.json"
    assert json.loads(fly_cp.read_text())["settings"]["class_balance_alpha"] == .75
    assert json.loads(plain_cp.read_text())["settings"]["bias_l2_multiplier"] == 4
    cache_file = tmp_path / "male-cns-fake.npz"
    np.savez_compressed(
        cache_file, signal=shared,
        timestamps=np.array(history.timestamps, dtype=np.int64),
        fingerprint=np.array(
            f"{FLY_CACHE_VERSION}:{source_hash}:{circuit_sha}:32"
        ),
    )
    recent = sample(95, start=history.timestamps[-1] + 300_000)
    forward_root = tmp_path / "forward"
    for arm, cp, signal in (
        ("with_fly", fly_cp, np.zeros(recent.size, dtype=np.float32)),
        ("without_fly", plain_cp, None),
        ("with_fly_signal_masked", fly_cp, None),
    ):
        frozen_forward_arm(
            checkpoint_file=cp, data=recent,
            features=market_features(recent, signal),
            output=forward_root / arm,
            after_timestamp_ms=history.timestamps[-1],
            source_csv_sha256="same-new-market-hash",
            name=arm, expect_fly=arm != "without_fly",
        )
    diag = investigate(
        history=history_file, with_checkpoint=fly_cp,
        without_checkpoint=plain_cp, forward_root=forward_root,
        fly_cache=cache_file, output=tmp_path / "offline-diagnostic.json",
    )
    assert diag["historical_neural_signal"]["cache_fingerprint_verified"]
    assert len(diag["historical_directional_regimes"]) == 7
    assert 0 < diag["development_only_up_prior"] < 1
    assert len(diag["fly_checkpoint_intercepts"]["preselected"]) <= 5
    assert diag["paired_same_frozen_weights"]["original_finalists"]["paired_decision_opportunities_correlated_across_agents"] >= 0
    assert diag["checkpoint_selection_not_modified"]
    assert (tmp_path / "offline-diagnostic.json").is_file()
    with pytest.raises(FileExistsError):
        investigate(
            history=history_file, with_checkpoint=fly_cp,
            without_checkpoint=plain_cp, forward_root=forward_root,
            output=tmp_path / "offline-diagnostic.json",
        )
    wrong = tmp_path / "wrong-cache.npz"
    np.savez_compressed(
        wrong, signal=shared,
        timestamps=np.array(history.timestamps),
        fingerprint=np.array("wrong-circuit"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        investigate(
            history=history_file, with_checkpoint=fly_cp,
            without_checkpoint=plain_cp, forward_root=forward_root,
            fly_cache=wrong,
        )
