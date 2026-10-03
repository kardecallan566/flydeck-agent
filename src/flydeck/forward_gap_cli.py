"""Repair a missing, bounded interval between two ALREADY downloaded 5m files.

Fails closed on Binance missing rows, out-of-order data, open candles,
timestamp overlap or existing output. Does NOT edit either original CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from .data.binance_provider import BinanceMarketDataProvider
from .data.market_cache import read_csv, write_csv
from .data.market_data import MarketCandle, dataset_from_candles

INTERVAL_MS = 300_000


class BoundedProvider(Protocol):
    def fetch_range(
        self, symbol: str, interval: str, start_ms: int, stop_ms: int,
    ) -> tuple[MarketCandle, ...]: ...


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def fill_closed_gap(
    *, base: Path, later: Path, output: Path,
    provider: BoundedProvider | None = None,
    now_ms: int | None = None,
    maximum_gap: int = 10_000,
) -> dict:
    """Fetch only missing timestamps in [old.end+5m, next.start)."""
    base, later, output = Path(base), Path(later), Path(output)
    manifest = output.with_suffix(".gap.json")
    if not base.is_file() or not later.is_file():
        raise FileNotFoundError("both source candle CSV files are required")
    if output.resolve() in {base.resolve(), later.resolve()}:
        raise ValueError("--output must differ from both source files")
    if output.exists() or manifest.exists():
        raise FileExistsError("gap output or its manifest already exists; refusing overwrite")
    previous = read_csv(base, symbol="BNBUSDT", interval="5m")
    following = read_csv(later, symbol="BNBUSDT", interval="5m")
    first = previous.candles[-1].timestamp + INTERVAL_MS
    stop = following.candles[0].timestamp
    if stop < first:
        raise ValueError("files overlap or are out of order: cannot repair as a gap")
    if stop == first:
        raise ValueError("no gap between these files: append them directly")
    if (stop - first) % INTERVAL_MS:
        raise ValueError("source boundaries are not aligned to exact five-minute candles")
    missing = (stop - first) // INTERVAL_MS
    if not 0 < missing <= maximum_gap:
        raise ValueError(f"gap of {missing} candles exceeds bounded limit {maximum_gap}")
    clock = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if stop > clock - INTERVAL_MS:
        raise ValueError("the later CSV begins too recently to confirm all gap candles closed")
    client = provider if provider is not None else BinanceMarketDataProvider()
    candles = client.fetch_range("BNBUSDT", "5m", first, stop)
    required = tuple(range(first, stop, INTERVAL_MS))
    actual = tuple(c.timestamp for c in candles)
    if actual != required:
        actual_set = set(actual)
        absent = [t for t in required if t not in actual_set]
        raise ValueError(
            f"Binance returned {len(actual)}/{len(required)} required gap candles; "
            f"first missing open={_utc(absent[0]) if absent else 'duplicate/out-of-order'}. "
            "No CSV was written."
        )
    repaired = dataset_from_candles(
        candles, symbol="BNBUSDT", interval="5m",
        interval_ms=INTERVAL_MS, source="exact-bounded-binance-gap-repair",
    )
    result = {
        "schema_version": 1, "research_only": True,
        "gap_fill_not_independent_holdout": True,
        "base_path": str(base), "base_sha256": _hash(base),
        "later_path": str(later), "later_sha256": _hash(later),
        "missing_candles_repaired": missing,
        "first_gap_open_ms": first, "last_gap_open_ms": stop - INTERVAL_MS,
        "first_gap_open_utc": _utc(first),
        "last_gap_closed_utc": _utc(stop),
        "expected_next_open_ms": stop,
        "source": "Binance public historic closed klines by start/end bounds",
    }
    # No source file is modified. Write CSV only after the exact grid passes.
    write_csv(repaired, output)
    result["gap_sha256"] = _hash(output)
    try:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=manifest.parent,
            suffix=".tmp", delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            json.dump(result, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
        os.replace(temp_path, manifest)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ONLY the exact missing 5m candles between two local CSVs."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        r = fill_closed_gap(base=args.base, later=args.later, output=args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Gap repair aborted: {exc}\n")
    print(
        f"Recovered exactly {r['missing_candles_repaired']} strictly closed "
        f"5m candles from {r['first_gap_open_utc']} to "
        f"{r['last_gap_closed_utc']} (UTC)."
    )
    print(f"Gap file: {args.output}; provenance: {args.output.with_suffix('.gap.json')}")
    print("Both source files are unchanged. Append base + gap, then + later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
