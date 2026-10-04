"""Strict prospective collection stays separate from inspected v3 cumulative replay."""
import pytest

from flydeck.data.market_cache import read_csv, write_csv
from flydeck.data.market_data import MarketCandle, dataset_from_candles
from flydeck.sealed_collect_cli import collect


def mk(path, n, start):
    candles = tuple(MarketCandle(
        timestamp=start + i * 300_000, open=100, high=101, low=99,
        close=100 + .0001*i, volume=1,
    ) for i in range(n))
    write_csv(dataset_from_candles(
        candles, symbol="BNBUSDT", interval="5m", source="test",
        interval_ms=300_000,
    ), path)


def test_partial_chain_reaches_exact_2000_without_scoring_or_dropping(tmp_path):
    p1, p2, p3 = (tmp_path / f"batch{i}.csv" for i in range(1, 4))
    mk(p1, 79, 1_200_000)
    mk(p2, 921, 1_200_000 + 79 * 300_000)
    mk(p3, 1000, 1_200_000 + 1000 * 300_000)
    half = tmp_path / "half.csv"
    r1 = collect(p1, p2, half)
    assert r1["count"] == 1000
    assert not r1["target_reached"]
    all_ = tmp_path / "all.csv"
    r2 = collect(half, p3, all_)
    assert r2["count"] == 2000 and r2["target_reached"]
    assert len(r2["source_chain"]) == 3
    assert r2["not_scored_or_trained_by_this_command"]
    ds = read_csv(all_, symbol="BNBUSDT", interval="5m")
    assert len(ds.candles) == 2000
    with pytest.raises(FileExistsError):
        collect(p1, p2, half)


def test_gap_overlap_and_over_target_fail_without_output(tmp_path):
    first, gap, overlap = (tmp_path / f"{name}.csv" for name in ("first", "gap", "overlap"))
    mk(first, 79, 1_200_000)
    mk(gap, 80, 1_200_000 + 80*300_000)
    mk(overlap, 80, 1_200_000 + 78*300_000)
    for later in (gap, overlap):
        out = tmp_path / f"bad-{later.stem}.csv"
        with pytest.raises(ValueError, match="nonconsecutive"):
            collect(first, later, out)
        assert not out.exists()
    proper = tmp_path / "proper.csv"
    mk(proper, 80, 1_200_000 + 79*300_000)
    with pytest.raises(ValueError, match="exceeds"):
        collect(first, proper, tmp_path / "too-many.csv", expected_total=100)


def test_existing_manifest_fingerprint_mismatch_fails(tmp_path):
    old, later = tmp_path / "old.csv", tmp_path / "later.csv"
    mk(old, 79, 1_200_000)
    mk(later, 80, 1_200_000 + 79*300_000)
    half = tmp_path / "half.csv"
    collect(old, later, half)
    mk(tmp_path / "more.csv", 30, 1_200_000 + 159*300_000)
    half.with_suffix(".sealed.json").write_text('{"mode":"collection_only","schema_version":1,"combined_sha256":"wrong"}')
    with pytest.raises(ValueError, match="seal provenance"):
        collect(half, tmp_path / "more.csv", tmp_path / "bad.csv")
