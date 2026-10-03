"""Append strictly-new closed 5m BNB candles without losing causal warmup.

Incremental downloads are accumulated as a *previously inspected* rolling
research stream. This stream can be replayed from the ORIGINAL frozen model
checkpoint without retraining. An ever-growing replay is NOT a new sealed test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data.market_cache import read_csv, write_csv
from .data.market_data import dataset_from_candles

INTERVAL_MS = 300_000


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def append_closed_candles(
    base: Path, new: Path, output: Path, *,
    manifest: Path | None = None,
) -> dict:
    """Validate old/new boundary, preserve EVERY candle, atomically write a copy.

    The base may be the 95-candle initial smoke file or an earlier cumulative
    file, but the next download must begin exactly one candle after it ends.
    Refuses an existing output and NEVER silently replaces source datasets.
    """
    base = Path(base)
    new = Path(new)
    output = Path(output)
    if not base.is_file() or not new.is_file():
        raise FileNotFoundError("both --base and --new must be existing 5m CSV files")
    if output.resolve() in {base.resolve(), new.resolve()}:
        raise ValueError("--output must differ from both source files")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cumulative CSV: {output}")
    if manifest is None:
        manifest = output.with_suffix(".manifest.json")
    manifest = Path(manifest)
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")
    left = read_csv(base, symbol="BNBUSDT", interval="5m")
    right = read_csv(new, symbol="BNBUSDT", interval="5m")
    if right.candles[0].timestamp != left.candles[-1].timestamp + INTERVAL_MS:
        raise ValueError(
            "overlap or gap between files: previous last open "
            f"{left.candles[-1].timestamp}, next first open "
            f"{right.candles[0].timestamp}; expected "
            f"{left.candles[-1].timestamp + INTERVAL_MS}. "
            "No data written."
        )
    combined = dataset_from_candles(
        left.candles + right.candles, symbol="BNBUSDT", interval="5m",
        source="cumulative-already-inspected-forward-research",
        interval_ms=INTERVAL_MS,
    )
    report = {
        "schema_version": 1,
        "research_only": True,
        "cumulative_already_inspected": True,
        "not_a_new_independent_holdout": True,
        "base_path": str(base), "new_path": str(new),
        "base_sha256": _hash(base), "new_sha256": _hash(new),
        "previous_rows": len(left.candles),
        "new_rows": len(right.candles),
        "total_rows": len(combined.candles),
        "first_open_ms": combined.candles[0].timestamp,
        "last_open_ms": combined.candles[-1].timestamp,
        "first_new_open_ms": right.candles[0].timestamp,
        "expected_labelled_decisions": max(0, len(combined.candles) - 33),
        "causal_warmup_once": 32,
        "replay_frozen_models_without_resume_state": True,
    }
    # write_csv already writes atomically. Publish the manifest only after
    # the combined market data passes the full canonical dataset validation.
    write_csv(combined, output)
    report["cumulative_sha256"] = _hash(output)
    try:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=manifest.parent,
            suffix=".tmp", delete=False,
        ) as temp:
            tmp = Path(temp.name)
            json.dump(report, temp, indent=2, ensure_ascii=False)
            temp.write("\n")
        os.replace(tmp, manifest)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append a truly NEW 5m batch to an inspected forward stream; no retraining/bets."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        r = append_closed_candles(args.base, args.new, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Append aborted: {exc}\n")
    print(f"Accumulated {r['total_rows']} continuous candles "
          f"({r['new_rows']} new). CSV: {args.output}")
    print("One-time causal warmup: 32 candles. "
          "REPLAY frozen models from scratch WITHOUT --resume-root.")
    print("This cumulative stream is ALREADY INSPECTED, not a sealed holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
