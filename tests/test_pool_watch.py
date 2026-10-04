"""Offline data-source and small-capital cost guardrails."""
import csv
import json
import pytest

from flydeck.pool_watch_cli import analyze, export, html_report


def item(key, symbol, apy, *, tvl=500000, project="pancakeswap-amm", chain="BSC"):
    return {"pool": key, "symbol": symbol, "apy": apy, "tvlUsd": tvl,
            "project": project, "chain": chain, "apyBase": apy, "apyReward": 0}


def data():
    return {"status": "success", "data": [
        item("stable", "USDT-USDC", 10),
        item("v3", "CAKE-WBNB", 120, project="pancakeswap-amm-v3"),
        item("thin", "BTC-USDC", 900, tvl=20),
        item("foreign", "USDT-USDC", 30, project="other"),
        item("html", "<script>alert(1)</script>-USDT", 5),
    ]}


def test_small_budget_filters_and_calculation():
    report = analyze(data())
    assert report["matching_pools"] == 3
    assert report["reserved_usd"] == 10
    assert report["apy_needed_to_cover_costs_pct"] == pytest.approx(121.6667)
    stable = report["pools"][0]
    assert stable["symbol"] == "USDT-USDC"
    assert stable["stable_pair_symbol_only"]
    assert stable["gross_minus_cost_usd"] < 0
    v3 = next(p for p in report["pools"] if p["version"] == "V3")
    assert v3["status"] == "V3_NOT_PERSONAL_YIELD"
    assert any("NÃO" in w for w in v3["warnings"])


def test_export_html_escapes_provider_text(tmp_path):
    report = analyze(data())
    rendered = html_report(report)
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    root = tmp_path / "new"
    export(report, root)
    assert (root / "report.html").exists()
    with (root / "pools.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert json.loads((root / "report.json").read_text())["research_only"]
    with pytest.raises(FileExistsError):
        export(report, root)


def test_empty_feed_can_fail_safe_without_picking_high_apy():
    result = analyze({"status": "success", "data": [item("bad", "x-y", 250, tvl=2)]})
    assert not result["pools"]
    assert "Nenhuma pool" in html_report(result)


@pytest.mark.parametrize("kwargs", [
    {"allocation": 30}, {"budget": -1}, {"cost": -1},
    {"min_tvl": 1}, {"days": 0}, {"limit": 0},
])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        analyze(data(), **kwargs)


def test_invalid_or_malicious_feed():
    for payload in ({}, {"status": "error", "data": []},
                    {"status": "success", "data": {}}):
        with pytest.raises(ValueError):
            analyze(payload)
    assert analyze({"status": "success", "data": [
        item("nan", "x-y", float("nan")),
        item("clean", "USDC-DAI", 12), item("clean", "USDC-DAI", 12),
    ]})["matching_pools"] == 1
