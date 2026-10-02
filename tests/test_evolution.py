"""Fast deterministic smoke tests; no downloads, wallet or connectome needed."""
from pathlib import Path
import numpy as np
import pytest

from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.evolution import (
    EvolutionSettings, initial_population, market_features, run_evolution, _labels,
)


def dataset(n=350, start=1_000_000):
    t = tuple(start + 300_000 * i for i in range(n))
    closes = tuple(100 + .005 * i + .25 * np.sin(i * .7) for i in range(n))
    return BNBPredictionDataset(
        timestamps=t, opens=closes,
        highs=tuple(c + .1 for c in closes),
        lows=tuple(c - .1 for c in closes),
        closes=closes, volumes=tuple(10 + i % 13 for i in range(n)),
    )


def test_features_are_causal_and_shared():
    data = dataset()
    one = market_features(data)
    modified = dataset()
    from dataclasses import replace
    prices = list(modified.closes)
    prices[-1] += 1000
    modified = replace(modified, closes=tuple(prices))
    two = market_features(modified)
    assert one.shape == (350, 13)
    assert not one.flags.writeable
    np.testing.assert_allclose(one[:-1], two[:-1])
    assert np.all(np.isfinite(one))


def test_population_is_reproducible_and_independent():
    config = EvolutionSettings(population=100, block_size=40, min_entries=1)
    one = initial_population(config, fly_available=False)
    two = initial_population(config, fly_available=False)
    assert len(one) == len(two) == 100
    assert len({a.agent_id for a in one}) == 100
    np.testing.assert_allclose(one[0].weights, two[0].weights)
    one[0].weights[0] = 99
    assert two[0].weights[0] != 99


def test_labels_are_next_candle_and_last_is_excluded():
    data = dataset()
    y, ready, _ = _labels(data, None)
    assert len(y) == data.size
    assert y[-1] == -1
    assert ready[33] == 34
    assert int(y[33]) == int(data.closes[34] > data.closes[33])


def test_end_to_end_reports_and_no_real_execution(tmp_path: Path):
    data = dataset(360)
    settings = EvolutionSettings(population=100, block_size=40, min_entries=1)
    output = tmp_path / "run"
    summary = run_evolution(data, settings, output=output)
    import csv, json
    with (output / "final_population.csv").open(encoding="utf-8") as f:
        final = list(csv.DictReader(f))
    assert len(final) == 100
    assert len(list(csv.DictReader((output / "all_block_results.csv").open()))) == 700
    assert (output / "lineage.csv").exists()
    assert (output / "finalists.json").exists()
    assert summary["real_execution_enabled"] is False
    assert json.loads((output / "summary.json").read_text())["population"] == 100


def test_reject_overlapping_recent_holdout(tmp_path):
    history = dataset(360)
    recent = dataset(150)
    config = EvolutionSettings(population=100, block_size=40, min_entries=1)
    with pytest.raises(ValueError, match="overlaps"):
        run_evolution(
            history, config, output=tmp_path / "overlap",
            recent=recent, recent_holdout=100,
        )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        EvolutionSettings(population=301)
    with pytest.raises(ValueError):
        EvolutionSettings(stake_fraction=1.0)


def test_separate_survival_risk_profit_and_evidence(tmp_path):
    import csv
    s = EvolutionSettings(population=100, block_size=40, min_entries=1)
    run_evolution(dataset(360), s, output=tmp_path)
    with (tmp_path / "all_block_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 700
    for row in rows:
        risk = row["risk_survived"] == "True"
        profitable = row["profitable"] == "True"
        survived = row["survived"] == "True"
        assert survived == (risk and profitable)
        assert not row["statistical_evidence"] == "True" or survived


def test_full_2000_closed_recent_holdout_without_adaptation(tmp_path):
    historic = dataset(360)
    latest = dataset(2000, start=historic.timestamps[-1] + 300000)
    s = EvolutionSettings(population=100, block_size=40, min_entries=1)
    report = run_evolution(
        historic, s, output=tmp_path / "full_holdout",
        recent=latest, recent_holdout=2000,
    )
    sealed = report["recent_test"]
    assert sealed["adaptation_candles"] == 0
    assert sealed["sealed_candles_requested"] == 2000
    assert sealed["warmup_skipped"] == 32
    assert (tmp_path / "full_holdout" / "recent_sealed_results.csv").exists()


def test_controlled_comparison_report_without_live_approval(tmp_path):
    from flydeck.evolution_comparison import write_ablation_comparison
    from flydeck.evolution import FAMILIES
    history = dataset(360)
    s = EvolutionSettings(population=100, block_size=40, min_entries=1)
    run_evolution(
        history, s, output=tmp_path / "with_fly",
        fly_signal=np.zeros(history.size, dtype=np.float32),
    )
    run_evolution(history, s, output=tmp_path / "without_fly")
    report = write_ablation_comparison(tmp_path)
    assert report["paired_seed_and_time_windows"]
    assert len(FAMILIES) == 10
    assert report["with_fly"]["all_validation"]["agents"] == 100
    assert (tmp_path / "ablation_report.csv").exists()
