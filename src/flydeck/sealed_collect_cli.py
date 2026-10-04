"""Append-only *collection* of never-scored prospective candles.

The collector does not inspect outcomes, train, predict, place trades or imply
that any given dataset is still independently unseen outside this command.
Never use a cumulative-diagnostic manifest to represent a sealed collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data.market_cache import read_csv, write_csv
from .data.market_data import dataset_from_candles

INTERVAL_MS = 300_000


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(
    base: Path, new: Path, output: Path, *,
    expected_total: int = 2000, max_rows: int = 10000,
) -> dict:
    base, new, output = Path(base), Path(new), Path(output)
    manifest = output.with_suffix(".sealed.json")
    if not 1 <= expected_total <= max_rows <= 100000:
        raise ValueError("invalid total/cap limits")
    if output.resolve() in {base.resolve(), new.resolve()}:
        raise ValueError("output may not be a source file")
    if output.exists() or manifest.exists():
        raise FileExistsError("output or seal manifest exists: no files overwritten")
    left = read_csv(base, symbol="BNBUSDT", interval="5m")
    right = read_csv(new, symbol="BNBUSDT", interval="5m")
    if right.candles[0].timestamp != left.candles[-1].timestamp + INTERVAL_MS:
        raise ValueError(
            "nonconsecutive 5m boundary: overlap or missing historical candles. "
            "Use exact bounded gap repair, not silent stitching."
        )
    n = len(left.candles) + len(right.candles)
    if n > max_rows or n > expected_total:
        raise ValueError(
            f"combined {n} candles exceeds declared {expected_total}-candle "
            "evaluation target: prevent silent retrospective cherry-picking"
        )
    ds = dataset_from_candles(
        left.candles + right.candles, symbol="BNBUSDT", interval="5m",
        interval_ms=INTERVAL_MS, source="sealed-research-collection-no-evaluation",
    )
    chain = []
    prior_manifest = base.with_suffix(".sealed.json")
    if prior_manifest.exists():
        previous = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if (
            previous.get("schema_version") != 1
            or previous.get("mode") != "collection_only"
            or previous.get("combined_sha256") != _hash(base)
            or previous.get("last_open_ms") != left.candles[-1].timestamp
            or previous.get("count") != len(left.candles)
            or previous.get("expected_total") != expected_total
        ):
            raise ValueError("existing seal provenance does not match previous CSV")
        chain = list(previous["source_chain"])
    else:
        chain = [{
            "source_csv": str(base),
            "sha256": _hash(base),
            "first_open_ms": left.candles[0].timestamp,
            "last_open_ms": left.candles[-1].timestamp,
            "count": len(left.candles),
        }]
    chain.append({
        "source_csv": str(new), "sha256": _hash(new),
        "first_open_ms": right.candles[0].timestamp,
        "last_open_ms": right.candles[-1].timestamp,
        "count": len(right.candles),
    })
    report = {
        "schema_version": 1, "mode": "collection_only", "research_only": True,
        "not_scored_or_trained_by_this_command": True,
        "cannot_prove_unseen_outside_this_command": True,
        "requires_new_independent_holdout_after_any_research_driven_changes": True,
        "count": n, "expected_total": expected_total,
        "target_reached": n == expected_total,
        "first_open_ms": ds.candles[0].timestamp,
        "last_open_ms": ds.candles[-1].timestamp,
        "source_chain": chain,
    }
    write_csv(ds, output)
    report["combined_sha256"] = _hash(output)
    try:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=manifest.parent,
            suffix=".tmp", delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(report, tmp, ensure_ascii=False, indent=2)
            tmp.write("\n")
        os.replace(tmp_path, manifest)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge strictly adjacent untouched 5m batches without inspecting outcomes."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=2000)
    args = parser.parse_args()
    try:
        result = collect(
            args.base, args.new, args.output, expected_total=args.target_count,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Sealed collection aborted: {exc}\n")
    print(f"Collected {result['count']}/{result['expected_total']} contiguous closed candles.")
    print(f"SHA256: {result['combined_sha256']}")
    print("Collection did NOT score/train on these data; "
          "independence still depends on your separate research protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
