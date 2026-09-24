from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .market_data import MarketCandle, MarketDataProvider, MarketDataset, dataset_from_candles, interval_to_milliseconds
from .market_cache import default_cache_path, read_csv, write_csv


@dataclass(frozen=True, slots=True)
class MarketDataHealth:
    candles: int
    start_timestamp: int
    end_timestamp: int
    expected_interval_ms: int
    gaps: int

    @property
    def complete(self) -> bool:
        return self.gaps == 0


def inspect_dataset(dataset: MarketDataset) -> MarketDataHealth:
    interval_ms = interval_to_milliseconds(dataset.interval)
    timestamps = dataset.timestamps
    gaps = sum(
        1 for previous, current in zip(timestamps, timestamps[1:]) if current - previous != interval_ms
    )
    return MarketDataHealth(
        candles=dataset.size,
        start_timestamp=timestamps[0],
        end_timestamp=timestamps[-1],
        expected_interval_ms=interval_ms,
        gaps=gaps,
    )


class MarketDataService:
    """OpenStock-inspired orchestration layer: provider -> normalized data -> cache.

    The neural agent never talks directly to Binance. This keeps acquisition,
    persistence, replay, and validation independent from the brain.
    """

    def __init__(self, provider: MarketDataProvider, cache_root: str | Path = "data/cache") -> None:
        self.provider = provider
        self.cache_root = Path(cache_root)

    def fetch(self, symbol: str, interval: str, limit: int = 1000) -> MarketDataset:
        candles = self.provider.fetch(symbol, interval, limit=limit)
        return dataset_from_candles(
            candles,
            symbol=symbol,
            interval=interval,
            source=self.provider.name,
            interval_ms=interval_to_milliseconds(interval),
        )

    def refresh(self, symbol: str, interval: str, limit: int = 1000) -> MarketDataset:
        path = default_cache_path(self.cache_root, symbol, interval)
        fresh = self.fetch(symbol, interval, limit=limit)

        existing: tuple[MarketCandle, ...] = ()
        if path.exists():
            cached = read_csv(path, symbol=symbol, interval=interval)
            existing = cached.candles

        merged = {candle.timestamp: candle for candle in existing}
        merged.update({candle.timestamp: candle for candle in fresh.candles})
        dataset = dataset_from_candles(
            merged.values(),
            symbol=symbol,
            interval=interval,
            source=f"{self.provider.name}+cache",
            interval_ms=interval_to_milliseconds(interval),
        )
        write_csv(dataset, path)
        return dataset

    def load(self, path: str | Path, symbol: str, interval: str) -> MarketDataset:
        return read_csv(path, symbol=symbol, interval=interval)

    def load_or_fetch(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 1000,
        refresh: bool = False,
    ) -> MarketDataset:
        path = default_cache_path(self.cache_root, symbol, interval)
        if path.exists() and not refresh:
            return self.load(path, symbol, interval)
        return self.refresh(symbol, interval, limit=limit)
