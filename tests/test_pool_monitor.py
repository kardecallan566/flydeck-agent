"""Pure offline Pool Watch UX and inter-snapshot alert regressions."""
import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from flydeck.pool_monitor import compare_reports, render_alerts_html
from flydeck.pool_report import render
from flydeck.pool_watch_cli import analyze, export, main


def pool(key, symbol, apy, tvl, project="pancakeswap-amm"):
    return {
        "pool": key, "symbol": symbol, "apy": apy, "tvlUsd": tvl,
        "project": project, "chain": "BSC", "apyBase": apy,
        "apyReward": 0,
    }


def report(rows, *, limited=1):
    return analyze(
        {"status": "success", "data": rows},
        budget=20, allocation=10, cost=.60,
        days=30, min_tvl=100_000, limit=limited,
    )


def test_new_layout_highlights_actual_small_budget_decision():
    r = report([
        pool("a", "USDT-USDC", .56, 500000),
        pool("b", "CAKE-WBNB", 30, 500000, project="pancakeswap-amm-v3"),
    ], limited=2)
    html = render(r)
    assert "Nenhuma das pools V2 exibidas cobre os custos" in html
    assert "73.0% de APY" in html
    assert "NÃO INVESTIR TAMBÉM É UMA OPÇÃO" in html
    assert "USDT-USDC" in html
    assert "APY INDICATIVO" in html
    assert "US$ 0.60" in html or "US$ 0.6000" in html
    assert "Dados agregados" in html
    assert "<details>" in html
    assert "viewport" in html


def test_full_monitoring_index_detects_changes_beyond_top_1():
    old = report([
        pool("a", "USDT-USDC", 12, 800000),
        pool("b", "BNB-USDT", 10, 600000),
        pool("disappeared", "CAKE-USDT", 8, 500000),
    ])
    new = report([
        pool("a", "USDT-USDC", 5, 550000),
        pool("b", "BNB-USDT", 14, 600000),
        pool("new", "FDUSD-USDC", 7, 500000),
    ])
    old["collected_utc"] = "2026-10-01T10:00:00+00:00"
    new["collected_utc"] = "2026-10-02T10:00:00+00:00"
    assert len(old["pools"]) == len(new["pools"]) == 1
    assert len(old["monitoring_index"]) == len(new["monitoring_index"]) == 3
    result = compare_reports(old, new, apy_change_pp=2, tvl_drop_pct=20)
    assert result["comparable_full_universe"]
    assert result["attention_count"] == 3
    assert result["information_count"] == 2
    kinds = [e["event"] for e in result["events"]]
    assert kinds.count("APY_DROP") == 1
    assert kinds.count("APY_RISE_UNVERIFIED") == 1
    assert kinds.count("TVL_DROP") == 1
    assert kinds.count("ENTERED_SOURCE_FILTER") == 1
    assert kinds.count("LEFT_SOURCE_FILTER") == 1
    tvl = next(e for e in result["events"] if e["event"] == "TVL_DROP")
    assert tvl["tvl_change_pct"] == pytest.approx(-31.25)


def test_legacy_snapshot_uses_partial_coverage_not_claim_complete():
    old = report([pool("x", "USDC-USDT", 8, 800000)])
    new = report([pool("x", "USDC-USDT", 5, 750000)])
    old.pop("monitoring_index")
    old["collected_utc"] = "2026-10-01T00:00:00+00:00"
    new["collected_utc"] = "2026-10-02T00:00:00+00:00"
    result = compare_reports(old, new)
    assert not result["comparable_full_universe"]
    assert result["attention_count"] == 1
    assert "old top-N" in result["disclaimer"]


def test_alerts_are_html_escaped_and_exported(tmp_path):
    old = report([pool("id<script>", '<img src=x onerror="alert(1)">', 10, 500000)])
    new = report([pool("id<script>", '<img src=x onerror="alert(1)">', 2, 300000)])
    old["collected_utc"] = "2026-10-01T00:00:00+00:00"
    new["collected_utc"] = "2026-10-02T00:00:00+00:00"
    new["comparison"] = compare_reports(old, new)
    page = render_alerts_html(new["comparison"])
    assert '<img src=x onerror="alert(1)">' not in page
    assert "&lt;img" in page
    dest = export(new, tmp_path / "run")
    assert (dest / "changes.json").is_file()
    assert (dest / "changes.csv").is_file()
    assert (dest / "changes.html").is_file()
    assert "changes.html" in (dest / "report.html").read_text(encoding="utf-8")
    with (dest / "changes.csv").open(newline="", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_no_false_changes_in_same_pool_and_timestamp_guards():
    old = report([pool("x", "USDC-USDT", 10, 500000)])
    new = report([pool("x", "USDC-USDT", 11, 470000)])
    old["collected_utc"] = "2026-10-01T00:00:00+00:00"
    new["collected_utc"] = "2026-10-02T00:00:00+00:00"
    assert not compare_reports(old, new)["events"]
    old["collected_utc"] = new["collected_utc"]
    with pytest.raises(ValueError, match="equal collection"):
        compare_reports(old, new)
    old["collected_utc"] = "2026-10-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="different feeds"):
        compare_reports({**old, "source": "unknown"}, new)
    with pytest.raises(ValueError):
        compare_reports(old, new, tvl_drop_pct=float("nan"))


def test_cli_offline_snapshot_and_previous_report(tmp_path, monkeypatch):
    import sys
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps({"status": "success", "synthetic_fixture_only": True,
                                   "data": [pool("x", "USDT-USDC", 10, 500000)]}),
                       encoding="utf-8")
    old = analyze(json.loads(earlier.read_text(encoding="utf-8")), budget=20, allocation=10, cost=.6)
    old["collected_utc"] = "2026-09-01T00:00:00+00:00"
    old_path = tmp_path / "old-report.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    current = tmp_path / "current.json"
    current.write_text(json.dumps({"status": "success", "synthetic_fixture_only": True,
                                   "data": [pool("x", "USDT-USDC", 3, 300000)]}),
                       encoding="utf-8")
    out = tmp_path / "new-report"
    monkeypatch.setattr(sys, "argv", [
        "flydeck-pool-watch", "--snapshot", str(current),
        "--previous", str(old_path), "--out-dir", str(out),
        "--budget-usd", "20", "--allocation-usd", "10",
        "--roundtrip-cost-usd", ".6",
    ])
    assert main() == 0
    assert (out / "changes.html").exists()
    assert "DEMONSTRAÇÃO: DADOS INVENTADOS" in (out / "report.html").read_text(encoding="utf-8")
    assert json.loads((out / "changes.json").read_text(encoding="utf-8"))["attention_count"] == 2
