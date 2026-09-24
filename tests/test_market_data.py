from __future__ import annotations

from pathlib import Path

import pytest

from flydeck.data.market_cache import read_csv, write_csv
from flydeck.data.market_data import MarketCandle, dataset_from_candles
from flydeck.data.market_service import inspect_dataset


def candle(timestamp: int, close: float = 100.0) -> MarketCandle:
    return MarketCandle(timestamp, close, close + 1.0, close - 1.0, close, 10.0)


def test_dataset_validation_accepts_canonical_5m_spacing() -> None:
    dataset = dataset_from_candles(
        [candle(0), candle(300_000, 101.0), candle(600_000, 100.5)],
        symbol="BNBUSDT",
        interval="5m",
        source="test",
        interval_ms=300_000,
    )
    health = inspect_dataset(dataset)
    assert health.candles == 3
    assert health.gaps == 0
    assert health.complete


def test_dataset_validation_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="unexpected candle spacing"):
        dataset_from_candles(
            [candle(0), candle(600_000)],
            symbol="BNBUSDT",
            interval="5m",
            source="test",
            interval_ms=300_000,
        )


def test_csv_cache_round_trip_uses_canonical_schema(tmp_path: Path) -> None:
    dataset = dataset_from_candles(
        [candle(0), candle(300_000, 101.0)],
        symbol="BNBUSDT",
        interval="5m",
        source="test",
        interval_ms=300_000,
    )
    path = write_csv(dataset, tmp_path / "bnb.csv")
    loaded = read_csv(path, symbol="BNBUSDT", interval="5m")
    assert loaded.candles == dataset.candles


def test_csv_loader_accepts_legacy_open_time_header(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "open_time,open,high,low,close,volume\n"
        "0,100,101,99,100,10\n"
        "300000,100,102,99,101,11\n",
        encoding="utf-8",
    )
    loaded = read_csv(path, symbol="BNBUSDT", interval="5m")
    assert loaded.timestamps == (0, 300_000)
    assert loaded.closes == (100.0, 101.0)
