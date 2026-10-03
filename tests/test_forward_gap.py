"""Offline bounded Binance backfill and non-overlapping continuity regressions."""
import pytest

from flydeck.data.binance_provider import BinanceMarketDataProvider
from flydeck.data.market_cache import read_csv, write_csv
from flydeck.data.market_data import MarketCandle, dataset_from_candles
from flydeck.forward_append_cli import append_closed_candles
from flydeck.forward_gap_cli import fill_closed_gap
from flydeck.recent_candles_cli import download_recent_closed

STEP = 300_000


def candles(n, start=1_200_000):
    return tuple(
        MarketCandle(
            timestamp=start + i * STEP, open=100 + i * .01,
            high=101 + i * .01, low=99 + i * .01,
            close=100.2 + i * .01, volume=10 + i % 7,
        ) for i in range(n)
    )


def save(path, rows):
    obj = dataset_from_candles(
        tuple(rows), symbol="BNBUSDT", interval="5m",
        interval_ms=STEP, source="offline-gap-fixture",
    )
    write_csv(obj, path)


class RangeFake:
    def __init__(self, all_candles, omit=None):
        self.rows = all_candles
        self.omit = omit
        self.requests = []

    def fetch_range(self, symbol, interval, start_ms, stop_ms):
        self.requests.append((symbol, interval, start_ms, stop_ms))
        return tuple(
            c for c in self.rows if start_ms <= c.timestamp < stop_ms
            and c.timestamp != self.omit
        )


def test_exact_gap_105_rows_and_complete_two_step_append(tmp_path):
    allrows = candles(300)
    base, later = tmp_path / "first.csv", tmp_path / "second.csv"
    save(base, allrows[:95])
    save(later, allrows[200:300])
    provider = RangeFake(allrows)
    gap = tmp_path / "missing.csv"
    repaired = fill_closed_gap(
        base=base, later=later, output=gap, provider=provider,
        now_ms=allrows[-1].timestamp + STEP,
    )
    assert repaired["missing_candles_repaired"] == 105
    assert provider.requests == [
        ("BNBUSDT", "5m", allrows[95].timestamp, allrows[200].timestamp)
    ]
    assert gap.with_suffix(".gap.json").is_file()
    mid = tmp_path / "mid.csv"
    whole = tmp_path / "whole.csv"
    append_closed_candles(base, gap, mid)
    final = append_closed_candles(mid, later, whole)
    assert final["total_rows"] == 300
    actual = read_csv(whole, symbol="BNBUSDT", interval="5m")
    assert len(actual.candles) == 300
    assert actual.candles[-1].timestamp == allrows[-1].timestamp


def test_gap_missing_one_candle_writes_nothing(tmp_path):
    rows = candles(300)
    old, recent = tmp_path / "old.csv", tmp_path / "new.csv"
    save(old, rows[:95])
    save(recent, rows[200:])
    path = tmp_path / "gap.csv"
    with pytest.raises(ValueError, match="104/105"):
        fill_closed_gap(
            base=old, later=recent, output=path,
            provider=RangeFake(rows, omit=rows[150].timestamp),
            now_ms=rows[-1].timestamp + STEP,
        )
    assert not path.exists()
    assert not path.with_suffix(".gap.json").exists()


def test_gap_avoids_existing_output_or_wrong_order(tmp_path):
    rows = candles(300)
    old, recent = tmp_path / "old.csv", tmp_path / "new.csv"
    save(old, rows[:95]); save(recent, rows[200:])
    gap = tmp_path / "gap.csv"; gap.write_text("preexisting", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        fill_closed_gap(base=old, later=recent, output=gap, provider=RangeFake(rows))
    gap.unlink()
    with pytest.raises(ValueError, match="overlap or are out of order"):
        fill_closed_gap(base=recent, later=old, output=gap, provider=RangeFake(rows))


def test_binance_range_paginates_exact_requested_grid(monkeypatch):
    provider = BinanceMarketDataProvider(retries=0)
    rows = candles(2100)
    calls = []
    def mocked_request(params):
        calls.append(params)
        part = [
            r for r in rows
            if params["startTime"] <= r.timestamp <= params["endTime"]
        ][:params["limit"]]
        return [
            [r.timestamp, str(r.open), str(r.high), str(r.low),
             str(r.close), str(r.volume)] for r in part
        ]
    monkeypatch.setattr(provider, "_request", mocked_request)
    answer = provider.fetch_range(
        "BNBUSDT", "5m", rows[0].timestamp, rows[-1].timestamp + STEP,
    )
    assert answer == rows
    assert len(calls) == 3
    assert calls[0]["startTime"] == rows[0].timestamp
    assert calls[-1]["limit"] == 100


def test_latest_window_must_not_silently_skip_intermediate_candles(tmp_path):
    rows = candles(300)
    class LatestFake:
        def fetch(self, symbol, interval, limit):
            return rows[-limit:]
    base = tmp_path / "old.csv"
    save(base, rows[:95])
    dest = tmp_path / "recent.csv"
    with pytest.raises(ValueError, match="Gap after previous dataset: 105"):
        download_recent_closed(
            count=100, available=True, after_history=base,
            output=dest, provider=LatestFake(),
            now_ms=rows[-1].timestamp + STEP,
        )
    assert not dest.exists()
