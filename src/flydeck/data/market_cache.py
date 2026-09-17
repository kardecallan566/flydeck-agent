from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .market_data import MarketCandle, MarketDataset, cache_path, dataset_from_candles, interval_to_milliseconds


CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def write_csv(dataset: MarketDataset, path: str | Path) -> Path:
    """Persist normalized data atomically so interrupted downloads cannot corrupt the cache."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=destination.parent, delete=False, suffix=".tmp"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(CANONICAL_COLUMNS)
        for candle in dataset.candles:
            writer.writerow(
                [candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume]
            )
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def read_csv(
    path: str | Path,
    *,
    symbol: str,
    interval: str,
    source: str = "csv-cache",
) -> MarketDataset:
    """Read both FlyDeck's canonical CSV and legacy singular OHLCV headers."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"market CSV is empty: {source_path}")

    names = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
    timestamp_name = names.get("timestamp") or names.get("open_time")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - names.keys()
    if timestamp_name is None:
        missing.add("timestamp")
    if missing:
        raise ValueError(f"missing market CSV columns: {sorted(missing)}")

    candles = [
        MarketCandle(
            timestamp=int(float(row[timestamp_name])),
            open=float(row[names["open"]]),
            high=float(row[names["high"]]),
            low=float(row[names["low"]]),
            close=float(row[names["close"]]),
            volume=float(row[names["volume"]]),
        )
        for row in rows
    ]
    return dataset_from_candles(
        candles,
        symbol=symbol,
        interval=interval,
        source=source,
        interval_ms=interval_to_milliseconds(interval),
    )


def default_cache_path(root: str | Path, symbol: str, interval: str) -> Path:
    return cache_path(root, symbol, interval)
