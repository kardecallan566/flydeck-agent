from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .finance import Candle


BINANCE_SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
SUPPORTED_INTERVALS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


@dataclass(frozen=True, slots=True)
class RealMarketDataset:
    symbol: str
    interval: str
    candles: tuple[Candle, ...]
    timestamps_ms: tuple[int, ...]
    source: str

    @property
    def rows(self) -> int:
        return len(self.candles)

    @property
    def start_timestamp_ms(self) -> int:
        return self.timestamps_ms[0]

    @property
    def end_timestamp_ms(self) -> int:
        return self.timestamps_ms[-1]


def _parse_timestamp(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def load_ohlcv_csv(path: str | Path, *, symbol: str = "UNKNOWN", interval: str = "UNKNOWN") -> RealMarketDataset:
    """Load canonical OHLCV CSV data without third-party dependencies.

    Required columns: timestamp, open, high, low, close, volume.
    Timestamp may be epoch milliseconds or an ISO-8601 timestamp.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    candles: list[Candle] = []
    timestamps: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = _parse_timestamp(row["timestamp"])
                candle = Candle(
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid OHLCV row {row_number}") from exc
            if candle.low <= 0 or candle.high <= 0 or candle.open <= 0 or candle.close <= 0 or candle.volume < 0:
                raise ValueError(f"invalid OHLCV values on row {row_number}")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError(f"timestamps must be strictly increasing at row {row_number}")
            if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
                raise ValueError(f"OHLC bounds are inconsistent on row {row_number}")
            timestamps.append(timestamp)
            candles.append(candle)

    if len(candles) < 26:
        raise ValueError("real market dataset must contain at least 26 candles")
    return RealMarketDataset(symbol.upper(), interval, tuple(candles), tuple(timestamps), f"csv:{path}")


def save_ohlcv_csv(dataset: RealMarketDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
        for timestamp, candle in zip(dataset.timestamps_ms, dataset.candles):
            writer.writerow((timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume))


def download_binance_spot_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
    output: str | Path,
    *,
    limit: int = 1000,
    pause_seconds: float = 0.15,
) -> RealMarketDataset:
    """Download a long historical OHLCV series from Binance's public spot API.

    ``start`` and ``end`` accept ISO-8601 dates/timestamps. The endpoint is public
    market data; no API key is required. Results are persisted as canonical CSV.
    """
    symbol = symbol.upper()
    interval = interval.strip()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be >= 0")

    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end) if end else None
    if end_ms is not None and end_ms <= start_ms:
        raise ValueError("end must be after start")

    rows: list[tuple[int, Candle]] = []
    cursor = start_ms
    interval_ms = SUPPORTED_INTERVALS[interval]

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        request = Request(
            f"{BINANCE_SPOT_KLINES_URL}?{urlencode(params)}",
            headers={"User-Agent": "flydeck-agent/real-market-benchmark"},
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise ValueError(f"unexpected Binance response: {payload!r}")
        if not payload:
            break

        previous_timestamp = rows[-1][0] if rows else None
        for raw in payload:
            if len(raw) < 6:
                raise ValueError("Binance kline row has fewer than 6 fields")
            timestamp = int(raw[0])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue
            candle = Candle(float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]), float(raw[5]))
            rows.append((timestamp, candle))
            previous_timestamp = timestamp

        last_timestamp = int(payload[-1][0])
        if len(payload) < limit or (end_ms is not None and last_timestamp >= end_ms - interval_ms):
            break
        next_cursor = last_timestamp + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if pause_seconds:
            time.sleep(pause_seconds)

    if end_ms is not None:
        rows = [(timestamp, candle) for timestamp, candle in rows if timestamp <= end_ms]
    if len(rows) < 26:
        raise ValueError("download returned fewer than 26 candles")

    dataset = RealMarketDataset(
        symbol=symbol,
        interval=interval,
        candles=tuple(candle for _, candle in rows),
        timestamps_ms=tuple(timestamp for timestamp, _ in rows),
        source="binance-spot-api",
    )
    save_ohlcv_csv(dataset, output)
    return dataset
