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
    available: bool = False,
    min_count: int = 35,
) -> dict:
    """Fail closed on short replies, gaps, partial or stale candles, overlap.

    For reproducibility, the CSV is never silently merged with historical data.
    An explicit --after-history rejects even partial overlap. Atomic write occurs
    only after every check passes.
    """
    if not 35 <= count <= 10_000:
        raise ValueError("count must be between 35 and 10000")
    if available and not 35 <= min_count <= count:
        raise ValueError("--min-count must be between 35 and --count when using --available")
    if maximum_delay_intervals < 1:
        raise ValueError("maximum_delay_intervals must be positive")
    clock_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    source = provider if provider is not None else BinanceMarketDataProvider()
    history_last: int | None = None
    if after_history is not None:
        old = read_csv(after_history, symbol="BNBUSDT", interval="5m")
        history_last = old.candles[-1].timestamp

    # Two spare rows cover an open candle and the exchange-boundary timing.
    # Request the usual count+2, but filter by the *actual previous timestamp*
    # before selecting anything. Overlap must never silently enter a holdout.
    received = source.fetch("BNBUSDT", "5m", limit=count + 2)
    closed = {
        candle.timestamp: candle for candle in received
        if candle.timestamp + INTERVAL_MS <= clock_ms
    }
    closed_new = [
        closed[t] for t in sorted(closed)
        if history_last is None or t > history_last
    ]
    available_now = len(closed_new)
    if not closed:
        raise ValueError(
            "Binance did not return any fully closed candles. "
            "Check your connection, system clock, and exchange response."
        )

    newest_closed_open = max(closed)
    # The previous file's last row may be recent: calculate how many full
    # 5-minute intervals are available and WHEN the requested target ends.
    target_close_ms = (
        history_last + (count + 1) * INTERVAL_MS
        if history_last is not None else None
    )
    deadline = (
        f" Expected {count} new candles after {_utc(history_last + INTERVAL_MS)}"
        f" at {_utc(target_close_ms)} (UTC)."
        if target_close_ms is not None else ""
    )
    if history_last is not None and newest_closed_open <= history_last:
        raise ValueError(
            f"No new fully closed Binance candles after {_utc(history_last + INTERVAL_MS)}."
            + deadline
        )
    if not available and available_now < count:
        raise ValueError(
            f"Only {available_now} of the {count} requested closed candles are "
            "available AFTER the historical CSV; the other recent candles "
            "overlap the inspected history. No file was written."
            + deadline
            + " Use --available to download only the new candles already present"
            " (minimum 35 for a forward smoke test), or run again later."
        )
    if available and available_now < min_count:
        raise ValueError(
            f"Only {available_now} completely NEW closed candles are available "
            f"since your historical CSV; --min-count={min_count}. "
            "No file was written."
            + deadline
        )
    desired = min(count, available_now) if available else count
    selected = closed_new[-desired:]
    ds = dataset_from_candles(
        selected, symbol="BNBUSDT", interval="5m",
        source="binance-public-closed-holdout", interval_ms=INTERVAL_MS,
    )
    oldest = ds.candles[0].timestamp
    newest = ds.candles[-1].timestamp
    lag = clock_ms - (newest + INTERVAL_MS)
    if lag > maximum_delay_intervals * INTERVAL_MS:
        raise ValueError(
            f"Binance's newest closed candle is stale by {lag / 60_000:.1f} "
            "minutes; no file written. Check data endpoint and clock."
        )
    if lag < 0:
        raise ValueError("latest candle is still open")
    if history_last is not None and oldest <= history_last:
        # Defense in depth; source selection above already excludes overlap.
        raise ValueError("download overlaps historical candles")
    # Check the old-to-new boundary in BOTH regular and --available mode.
    # The prior implementation checked only a partial --available response.
    # If 100+ candles elapsed, selecting the last 100 silently skipped older
    # unseen candles even though every timestamp in the selected file was new.
    if history_last is not None and oldest != history_last + INTERVAL_MS:
        missing = (oldest - history_last) // INTERVAL_MS - 1
        raise ValueError(
            f"Gap after previous dataset: {missing} five-minute candles "
            f"were skipped between {_utc(history_last + INTERVAL_MS)} and "
            f"{_utc(oldest)}. No file was written. "
            "Use flydeck-fill-gap with your previous CSV and an already "
            "downloaded later CSV, or fetch a sufficiently large window "
            "that starts exactly at the old file's next five-minute candle."
        )
    result = {
        "symbol": "BNBUSDT", "interval": "5m",
        "count": len(selected), "requested_count": count,
        "available_after_history_in_requested_window": available_now,
        "partial_download_explicit": bool(available and len(selected) < count),
        "first_open_ms": oldest,
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
        "--after-history", type=Path,
        help="Only fully closed candles strictly newer than this CSV may be selected",
    )
    parser.add_argument(
        "--available", action="store_true",
        help="Explicitly accept fewer than --count if insufficient new candles exist",
    )
    parser.add_argument(
        "--min-count", type=int, default=35,
        help="Minimum new candles accepted with --available (default: 35)",
    )
    args = parser.parse_args()
    try:
        result = download_recent_closed(
            count=args.count, output=args.output, after_history=args.after_history,
            available=args.available, min_count=args.min_count,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Download aborted: {exc}\n")
    print(f"Downloaded {result['count']} CLOSED, nonoverlapping 5m candles to {args.output}")
    if result["partial_download_explicit"]:
        print(
            f"Partial new dataset: {result['count']}/{result['requested_count']} "
            "requested candles. Smoke test only; not a 2000-candle holdout."
        )
    print(f"UTC: {result['first_open_utc']} to {result['last_closed_utc']}")
    print("No API key, no wallet, no partially open candle. Metadata: "
          f"{args.output.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
