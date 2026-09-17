from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class MarketCandle:
    """Canonical OHLCV candle used by FlyDeck's data layer."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Validated, chronological market series independent of its source."""

    symbol: str
    interval: str
    candles: tuple[MarketCandle, ...]
    source: str

    @property
    def size(self) -> int:
        return len(self.candles)

    @property
    def timestamps(self) -> tuple[int, ...]:
        return tuple(c.timestamp for c in self.candles)

    @property
    def opens(self) -> tuple[float, ...]:
        return tuple(c.open for c in self.candles)

    @property
    def highs(self) -> tuple[float, ...]:
        return tuple(c.high for c in self.candles)

    @property
    def lows(self) -> tuple[float, ...]:
        return tuple(c.low for c in self.candles)

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(c.close for c in self.candles)

    @property
    def volumes(self) -> tuple[float, ...]:
        return tuple(c.volume for c in self.candles)


class MarketDataProvider(Protocol):
    """Source contract used by backtests, replay, and live acquisition."""

    name: str

    def fetch(self, symbol: str, interval: str, limit: int = 1000) -> Sequence[MarketCandle]:
        ...


def validate_candles(
    candles: Iterable[MarketCandle],
    *,
    interval_ms: int | None = None,
    require_positive_prices: bool = True,
) -> tuple[MarketCandle, ...]:
    """Validate and deduplicate candles without changing their chronological order."""

    ordered = sorted(candles, key=lambda candle: candle.timestamp)
    deduped: list[MarketCandle] = []
    seen: set[int] = set()

    for candle in ordered:
        if candle.timestamp in seen:
            continue
        seen.add(candle.timestamp)

        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if any(value != value for value in values):
            raise ValueError(f"non-finite candle at timestamp {candle.timestamp}")
        if require_positive_prices and min(candle.open, candle.high, candle.low, candle.close) <= 0:
            raise ValueError(f"non-positive price at timestamp {candle.timestamp}")
        if candle.high < max(candle.open, candle.close, candle.low):
            raise ValueError(f"invalid high at timestamp {candle.timestamp}")
        if candle.low > min(candle.open, candle.close, candle.high):
            raise ValueError(f"invalid low at timestamp {candle.timestamp}")
        if candle.volume < 0:
            raise ValueError(f"negative volume at timestamp {candle.timestamp}")

        if deduped and candle.timestamp <= deduped[-1].timestamp:
            raise ValueError("market data must be strictly chronological")
        if interval_ms is not None and deduped:
            delta = candle.timestamp - deduped[-1].timestamp
            if delta != interval_ms:
                raise ValueError(
                    f"unexpected candle spacing: expected {interval_ms}ms, got {delta}ms "
                    f"at timestamp {candle.timestamp}"
                )
        deduped.append(candle)

    return tuple(deduped)


def dataset_from_candles(
    candles: Iterable[MarketCandle],
    *,
    symbol: str,
    interval: str,
    source: str,
    interval_ms: int | None = None,
) -> MarketDataset:
    validated = validate_candles(candles, interval_ms=interval_ms)
    if not validated:
        raise ValueError("market dataset is empty")
    return MarketDataset(symbol=symbol, interval=interval, candles=validated, source=source)


def interval_to_milliseconds(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if len(interval) < 2 or interval[-1] not in units:
        raise ValueError(f"unsupported interval: {interval}")
    try:
        amount = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported interval: {interval}") from exc
    if amount <= 0:
        raise ValueError(f"unsupported interval: {interval}")
    return amount * units[interval[-1]]


def cache_path(root: str | Path, symbol: str, interval: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("-", "_")
    return Path(root) / f"{safe_symbol}_{interval}.csv"
