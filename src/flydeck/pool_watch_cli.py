"""Read-only PancakeSwap pool discovery for $5-20 budgets.

Uses PUBLIC third-party DefiLlama yields; NOT a PancakeSwap quote.
Reports cash-flow scenarios net of user-supplied round-trip costs, but
EXCLUDES market losses and V3 personal range effects. Never executes trades.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://yields.llama.fi/pools"
PROJECTS = {"pancakeswap-amm", "pancakeswap-amm-v3"}
CHAINS = {"BSC", "Binance"}
STABLE = {"USDC", "USDT", "DAI", "FDUSD", "TUSD"}
COLUMNS = ("symbol", "version", "tvl_usd", "reported_apy_pct",
           "gross_yield_usd", "gross_minus_cost_usd", "break_even_days",
           "status", "pool_url")


def _valid(v, *, minimum=0, maximum=100000):
    try:
        n = float(v)
    except (ValueError, TypeError):
        return None
    return n if math.isfinite(n) and minimum <= n <= maximum else None


def fetch_public_pools(timeout=25):
    req = Request(SOURCE, headers={
        "Accept": "application/json", "User-Agent": "FlyDeckReadOnlyResearch/0.1",
    })
    with urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("yield feed returned non-200 response")
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("yield feed over 32 MiB cap")
    return json.loads(data)


def analyze(payload, *, budget=20, allocation=10, cost=1, days=30,
            min_tvl=100000, limit=12):
    if not all(math.isfinite(v) for v in (budget, allocation, cost, min_tvl)):
        raise ValueError("budget/cost/TVL must be finite")
    if not 0 < allocation <= budget or cost < 0 or min_tvl < 10000:
        raise ValueError("invalid allocation/cost/minimum TVL")
    if not 1 <= days <= 3650 or not 1 <= limit <= 100:
        raise ValueError("days/limit out of range")
    if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
        raise ValueError("untrusted source response")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) > 200000:
        raise ValueError("expected bounded DefiLlama data list")
    output, seen = [], set()
    for p in rows:
        if not isinstance(p, dict) or p.get("project") not in PROJECTS or p.get("chain") not in CHAINS:
            continue
        apy, tvl = _valid(p.get("apy")), _valid(p.get("tvlUsd"), maximum=1e15)
        key, symbol = str(p.get("pool") or "")[:160], str(p.get("symbol") or "")[:80]
        if apy is None or tvl is None or tvl < min_tvl or not key or not symbol or key in seen:
            continue
        seen.add(key)
        v3 = p["project"] == "pancakeswap-amm-v3"
        gross = allocation * apy / 100 * days / 365
        annual = allocation * apy / 100
        tok = symbol.replace("/", "-").upper().split("-")
        stable_symbols = len(tok) == 2 and all(t in STABLE for t in tok)
        warnings = ["APY de agregador, não rendimento confirmado"]
        if v3:
            warnings.append("V3: renda pessoal depende da faixa ativa; APY agregado NÃO é cotação")
        if stable_symbols:
            warnings.append("Stablecoins também podem perder paridade")
        if apy >= 50:
            warnings.append("APY elevado: investigar risco do token, recompensas e sustentabilidade")
        output.append({
            "symbol": symbol, "version": "V3" if v3 else "V2",
            "pool_id": key, "tvl_usd": round(tvl, 2),
            "reported_apy_pct": round(apy, 5),
            "reported_base_apy_pct": _valid(p.get("apyBase")),
            "reported_reward_apy_pct": _valid(p.get("apyReward")),
            "stable_pair_symbol_only": stable_symbols,
            "gross_yield_usd": round(gross, 6),
            "gross_minus_cost_usd": round(gross - cost, 6),
            "break_even_days": math.ceil(365 * cost / annual) if annual else None,
            "status": (
                "V3_NOT_PERSONAL_YIELD" if v3 else
                "BELOW_ASSUMED_COSTS" if gross <= cost else
                "HIGH_APY_REVIEW" if apy >= 50 else
                "GROSS_ONLY_RISKS_NOT_INCLUDED"
            ),
            "warnings": warnings,
            "pool_url": "https://defillama.com/yields/pool/" + key,
            "verify_official_url": "https://pancakeswap.finance/liquidity/pools",
        })
    # Liquidity/quality research order, NEVER highest advertised APY first.
    output.sort(key=lambda p: (
        not p["stable_pair_symbol_only"], p["version"] != "V2",
        -p["tvl_usd"], p["reported_apy_pct"], p["pool_id"],
    ))
    return {
        "research_only": True, "third_party_not_official_quote": True,
        "synthetic_fixture_only": payload.get("synthetic_fixture_only") is True,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE, "budget_usd": budget, "allocation_usd": allocation,
        "reserved_usd": round(budget - allocation, 4),
        "assumed_roundtrip_cost_usd": cost, "days": days,
        "apy_needed_to_cover_costs_pct": round(100 * 365 * cost / (days * allocation), 4),
        "matching_pools": len(output), "pools": output[:limit],
        "min_tvl_usd": min_tvl,
        # Full SOURCE-FILTERED index permits meaningful monitoring even when
        # only the 12 highest-liquidity examples were displayed in HTML.
        "monitoring_index": [{
            "pool_id": p["pool_id"], "symbol": p["symbol"],
            "reported_apy_pct": p["reported_apy_pct"],
            "tvl_usd": p["tvl_usd"],
        } for p in output],
        "caveats": [
            "No position is always an option, particularly below $20.",
            "No live contract, actual withdrawability or actual user V3 APR verified.",
            "No slippage, price moves, impermanent loss, depeg, taxes or smart contract risk modeled.",
            "Farm rewards priced in volatile tokens can fall before harvest.",
            "Fees and yield shown are hypothetical, linear prorating of third-party APY.",
        ],
    }


def html_report(report):
    """Improved dashboard kept as a stable interface for callers/tests."""
    from .pool_report import render
    return render(report)


def export(report, output_dir):
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("output directory exists; refuse overwriting previous reports")
    root.mkdir(parents=True)
    try:
        (root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (root / "report.html").write_text(html_report(report), encoding="utf-8")
        if "comparison" in report:
            from .pool_monitor import save_comparison
            save_comparison(report["comparison"], root)
        fields = ("symbol", "version", "tvl_usd", "reported_apy_pct",
                  "gross_yield_usd", "gross_minus_cost_usd", "break_even_days",
                  "status", "pool_url")
        with (root / "pools.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pool in report["pools"]:
                writer.writerow({field: pool[field] for field in fields})
    except BaseException:
        for file in root.iterdir():
            file.unlink()
        root.rmdir()
        raise
    return root


def main():
    p = argparse.ArgumentParser(description="Read-only, public PancakeSwap BSC pool scanner; never trades.")
    p.add_argument("--budget-usd", type=float, default=20)
    p.add_argument("--allocation-usd", type=float, default=10)
    p.add_argument("--roundtrip-cost-usd", type=float, default=1)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--min-tvl-usd", type=float, default=100000)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--snapshot", type=Path, help="Offline DefiLlama response JSON; no network")
    p.add_argument(
        "--previous", type=Path,
        help="Earlier report.json for offline APY / liquidity change alerts",
    )
    p.add_argument(
        "--apy-change-pp", type=float, default=2.0,
        help="Notify of changes >= this many absolute APY percentage points",
    )
    p.add_argument(
        "--tvl-drop-pct", type=float, default=20.0,
        help="Notify when aggregated pool TVL falls by this percentage",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        source = json.loads(a.snapshot.read_text(encoding="utf-8")) if a.snapshot else fetch_public_pools()
        result = analyze(
            source, budget=a.budget_usd, allocation=a.allocation_usd,
            cost=a.roundtrip_cost_usd, days=a.days, min_tvl=a.min_tvl_usd,
            limit=a.limit,
        )
        result["input_mode"] = "offline_snapshot" if a.snapshot else "online_public_feed"
        if a.snapshot:
            result["offline_snapshot_file"] = str(a.snapshot)
        if a.previous is not None:
            from .pool_monitor import compare_reports
            previous = json.loads(a.previous.read_text(encoding="utf-8"))
            result["comparison"] = compare_reports(
                previous, result,
                apy_change_pp=a.apy_change_pp,
                tvl_drop_pct=a.tvl_drop_pct,
            )
        location = export(result, a.out_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        p.exit(2, f"Pool screen aborted: {exc}\n")
    print("PancakeSwap BSC pools passing source/TVL filters:", result["matching_pools"])
    print("Estimated APY needed JUST for assumed costs:",
          str(result["apy_needed_to_cover_costs_pct"]) + "%")
    print("Static report:", location / "report.html")
    print("CSV:", location / "pools.csv")
    if "comparison" in result:
        print("Source-change alerts:", location / "changes.html",
              "| attention:", result["comparison"]["attention_count"])
    print("This is NOT a live PancakeSwap quote or a trade recommendation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
