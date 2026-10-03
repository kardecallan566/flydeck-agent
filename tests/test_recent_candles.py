"""Mock-only unit tests for public Binance paginated closed-candle downloader."""
import pytest

from flydeck.data.market_data import MarketCandle
from flydeck.recent_candles_cli import INTERVAL_MS, download_recent_closed


class FakeProvider:
    def __init__(self, candles):
        self.candles = candles
        self.calls = []

    def fetch(self, symbol, interval, limit):
        self.calls.append((symbol, interval, limit))
        return tuple(self.candles[-limit:])


def rows(n=2050, start=1_000_000):
    return [
        MarketCandle(
            timestamp=start + i * INTERVAL_MS,
            open=100 + i / 1000,
            high=100.5 + i / 1000,
            low=99.5 + i / 1000,
            close=100 + i / 1000,
            volume=1 + i % 10,
        )
        for i in range(n)
    ]


def test_download_2000_discards_open_candle_and_paginates(tmp_path):
    candles = rows()
    provider = FakeProvider(candles)
    clock = candles[-2].timestamp + INTERVAL_MS + 10_000
    out = tmp_path / "new.csv"
    info = download_recent_closed(
        count=2000, output=out, provider=provider, now_ms=clock,
    )
    assert info["count"] == 2000
    assert info["last_open_ms"] == candles[-2].timestamp
    assert provider.calls == [("BNBUSDT", "5m", 2002)]
    assert out.exists() and out.with_suffix(".meta.json").exists()
    assert len(out.read_text().splitlines()) == 2001


def test_overlap_rejected_without_overwriting_file(tmp_path):
    from flydeck.data.market_data import dataset_from_candles
    from flydeck.data.market_cache import write_csv
    candles = rows()
    historic = dataset_from_candles(
        candles[:100], symbol="BNBUSDT", interval="5m",
        source="fixture", interval_ms=INTERVAL_MS,
    )
    old = tmp_path / "old.csv"
    write_csv(historic, old)
    # Recent count 2000 from 2050 starts at index 49, overlapping old.
    with pytest.raises(ValueError, match="overlap"):
        download_recent_closed(
            count=2000, output=tmp_path / "recent.csv",
            after_history=old, provider=FakeProvider(candles),
            now_ms=candles[-2].timestamp + INTERVAL_MS + 10_000,
        )
    assert not (tmp_path / "recent.csv").exists()


def test_missing_candle_detected(tmp_path):
    candles = rows()
    del candles[-200]
    with pytest.raises(ValueError, match="spacing"):
        download_recent_closed(
            count=2000, output=tmp_path / "bad.csv",
            provider=FakeProvider(candles),
            now_ms=candles[-2].timestamp + INTERVAL_MS + 10_000,
        )

def test_short_new_window_gives_counts_deadline_no_partial_file(tmp_path):
    from flydeck.data.market_data import dataset_from_candles
    from flydeck.data.market_cache import write_csv
    candles = rows(230)
    old = dataset_from_candles(
        candles[:180], symbol="BNBUSDT", interval="5m",
        source="fixture", interval_ms=INTERVAL_MS,
    )
    hist = tmp_path / "recent_old.csv"
    write_csv(old, hist)
    clock = candles[-2].timestamp + INTERVAL_MS + 1000
    with pytest.raises(ValueError, match=r"Only 49 of the 100 requested.*Expected 100"):
        download_recent_closed(
            count=100, output=tmp_path / "new.csv",
            after_history=hist, provider=FakeProvider(candles),
            now_ms=clock,
        )
    assert not (tmp_path / "new.csv").exists()


def test_explicit_available_download_never_overlaps_and_reports_partial(tmp_path):
    from flydeck.data.market_data import dataset_from_candles
    from flydeck.data.market_cache import write_csv, read_csv
    candles = rows(230)
    hist = tmp_path / "prior.csv"
    old = dataset_from_candles(
        candles[:180], symbol="BNBUSDT", interval="5m",
        source="fixture", interval_ms=INTERVAL_MS,
    )
    write_csv(old, hist)
    clock = candles[-2].timestamp + INTERVAL_MS + 1000
    path = tmp_path / "new.csv"
    info = download_recent_closed(
        count=100, output=path, after_history=hist,
        provider=FakeProvider(candles), now_ms=clock,
        available=True, min_count=35,
    )
    assert info["count"] == 49
    assert info["requested_count"] == 100
    assert info["available_after_history_in_requested_window"] == 49
    assert info["partial_download_explicit"]
    read = read_csv(path, symbol="BNBUSDT", interval="5m")
    assert read.candles[0].timestamp == candles[180].timestamp
    assert read.candles[-1].timestamp == candles[-2].timestamp
    assert read.candles[0].timestamp > old.candles[-1].timestamp
    assert len(read.candles) == 49


def test_available_rejects_fewer_than_min_count(tmp_path):
    from flydeck.data.market_data import dataset_from_candles
    from flydeck.data.market_cache import write_csv
    candles = rows(210)
    hist = tmp_path / "prior.csv"
    old = dataset_from_candles(
        candles[:180], symbol="BNBUSDT", interval="5m",
        source="fixture", interval_ms=INTERVAL_MS,
    )
    write_csv(old, hist)
    with pytest.raises(ValueError, match=r"Only 29.*min-count=35"):
        download_recent_closed(
            count=100, output=tmp_path / "new.csv", after_history=hist,
            provider=FakeProvider(candles),
            now_ms=candles[-2].timestamp + INTERVAL_MS + 1000,
            available=True,
        )
    assert not (tmp_path / "new.csv").exists()


def test_available_rejects_missing_first_new_candle(tmp_path):
    from flydeck.data.market_data import dataset_from_candles
    from flydeck.data.market_cache import write_csv
    candles = rows(230)
    hist = tmp_path / "prior.csv"
    old = dataset_from_candles(
        candles[:180], symbol="BNBUSDT", interval="5m",
        source="fixture", interval_ms=INTERVAL_MS,
    )
    write_csv(old, hist)
    del candles[180]
    with pytest.raises(ValueError, match="Gap after previous dataset"):
        download_recent_closed(
            count=100, output=tmp_path / "new.csv", after_history=hist,
            provider=FakeProvider(candles),
            now_ms=candles[-2].timestamp + INTERVAL_MS + 1000,
            available=True,
        )
    assert not (tmp_path / "new.csv").exists()
