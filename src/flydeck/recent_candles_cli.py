"""Download a strictly closed, chronological public BNB 5-minute holdout.

The Binance API supports 1,000 candles per page; the existing market provider
handles pagination. No API key, wallet, or private endpoints.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .data.binance_provider import BinanceMarketDataProvider
from .data.market_cache import read_csv, write_csv
from .data.market_data import MarketCandle, dataset_from_candles

INTERVAL_MS = 300_000


class CandleProvider(Protocol):
    def fetch(self, symbol: str, interval: str, limit: int) -> tuple[MarketCandle, ...]: ...


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def download_recent_closed(
    *, count: int = 2000, output: str | Path,
    after_history: str | Path | None = None,
    provider: CandleProvider | None = None,
    now_ms: int | None = None,
    maximum_delay_intervals: int = 3,
) -> dict:
    """Fail closed on short replies, gaps, partial or stale candles, overlap.

    For reproducibility, the CSV is never silently merged with historical data.
    An explicit --after-history rejects even partial overlap. Atomic write occurs
    only after every check passes.
    """
    if not 100 <= count <= 10_000:
        raise ValueError("count must be between 100 and 10000")
    if maximum_delay_intervals < 1:
        raise ValueError("maximum_delay_intervals must be positive")
    clock_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    source = provider if provider is not None else BinanceMarketDataProvider()
    # Two spare rows cover a partially opened latest candle and boundary drift.
    received = source.fetch("BNBUSDT", "5m", limit=count + 2)
    closed = {
        candle.timestamp: candle for candle in received
        if candle.timestamp + INTERVAL_MS <= clock_ms
    }
    if len(closed) < count:
        raise ValueError(f"Binance supplied only {len(closed)} closed candles; expected {count}")
    selected = [closed[t] for t in sorted(closed)[-count:]]
    ds = dataset_from_candles(
        selected, symbol="BNBUSDT", interval="5m",
        source="binance-public-closed-holdout", interval_ms=INTERVAL_MS,
    )
    oldest = ds.candles[0].timestamp
    newest = ds.candles[-1].timestamp
    lag = clock_ms - (newest + INTERVAL_MS)
    if lag > maximum_delay_intervals * INTERVAL_MS:
        raise ValueError(f"last closed candle is stale by {lag / 60_000:.1f} minutes")
    if lag < 0:
        raise ValueError("latest candle is still open")

    history_last: int | None = None
    if after_history is not None:
        old = read_csv(after_history, symbol="BNBUSDT", interval="5m")
        history_last = old.candles[-1].timestamp
        if oldest <= history_last:
            raise ValueError(
                "recent dataset overlaps historical CSV; collect a later holdout "
                "or choose fewer candles. Do not silently deduplicate sealed data."
            )
    result = {
        "symbol": "BNBUSDT", "interval": "5m",
        "count": len(selected), "first_open_ms": oldest,
        "last_open_ms": newest, "first_open_utc": _utc(oldest),
        "last_closed_utc": _utc(newest + INTERVAL_MS),
        "downloaded_utc": _utc(clock_ms),
        "history_last_open_ms": history_last,
        "no_partial_candles": True, "no_gaps": True,
        "strictly_newer_than_history": history_last is None or oldest > history_last,
    }
    destination = Path(output)
    write_csv(ds, destination)
    destination.with_suffix(".meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get latest 2000 closed public Binance BNBUSDT 5m candles; no API key."
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("data/cache/BNBUSDT_recent_5m.csv"),
    )
    parser.add_argument(
        "--after-history", type=Path, help="Fail if any downloaded candle overlaps this historical CSV",
    )
    args = parser.parse_args()
    result = download_recent_closed(
        count=args.count, output=args.output, after_history=args.after_history,
    )
    print(f"Downloaded {result['count']} CLOSED 5m candles to {args.output}")
    print(f"UTC: {result['first_open_utc']} to {result['last_closed_utc']}")
    print("No API key, no wallet, no partially open candle. Metadata: "
          f"{args.output.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
