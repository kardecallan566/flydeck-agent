"""Offline regressions for short-interval changes and LP-price scenarios."""
import csv
import json

import pytest

from flydeck.pool_monitor import compare_reports, render_alerts_html
from flydeck.pool_report import render
from flydeck.pool_stress import v2_stress_scenarios
from flydeck.pool_watch_cli import analyze, export


def source(apy=10.0, tvl=600000.0, *, version="pancakeswap-amm"):
    return {"status": "success", "data": [
        {"pool": "stable-id", "project": version, "chain": "BSC",
         "symbol": "USDT-WBNB", "apy": apy, "tvlUsd": tvl,
         "apyBase": apy, "apyReward": 0}
    ]}


def report(apy=10.0, tvl=600000.0, **kwargs):
    return analyze(
        source(apy, tvl), budget=20, allocation=10, cost=.6,
        days=30, min_tvl=100000, **kwargs
    )


def test_one_minute_apart_and_below_alert_threshold_are_visible():
    old = report(apy=1.30, tvl=600000)
    now = report(apy=1.35, tvl=599000)
    old["collected_utc"] = "2026-10-04T04:49:44+00:00"
    now["collected_utc"] = "2026-10-04T04:50:07+00:00"
    compared = compare_reports(old, now)
    assert compared["events"] == []
    assert compared["attention_count"] == 0
    assert compared["minor_changes"] == 1
    assert compared["matching_pools"] == 1
    assert compared["short_observation_window"] is True
    assert compared["interval_minutes"] == pytest.approx(23 / 60, abs=.001)
    assert compared["top_movements"][0]["apy_change_pp"] == pytest.approx(.05)
    page = render_alerts_html(compared)
    assert "Maiores variações observadas" in page
    assert "USDT-WBNB" in page
    assert "menos de uma hora" in page
    assert "+0.050 p.p." in page
    assert "NENHUMA" not in page  # no false firm assertion of a stable market


def test_reverse_timestamp_prevented_and_old_index_still_supported():
    old = report()
    now = report(apy=12)
    old["collected_utc"] = "2026-10-05T04:00:00+00:00"
    now["collected_utc"] = "2026-10-04T04:00:00+00:00"
    with pytest.raises(ValueError, match="strictly later"):
        compare_reports(old, now)
    now["collected_utc"] = "2026-10-06T04:00:00+00:00"
    old.pop("monitoring_index")
    diag = compare_reports(old, now)
    assert diag["prior_complete_index"] is False
    assert diag["comparable_full_universe"] is False


def test_hypothetical_v2_stress_mathematics_and_zero_shock():
    stress = v2_stress_scenarios(
        capital_usd=10, hypothetical_gross_yield_usd=.082,
        roundtrip_cost_usd=.60, shock_pct=30
    )
    assert stress["model_is_not_v3"]
    assert len(stress["rows"]) == 5
    zero = next(r for r in stress["rows"] if r["token_a_price_change_pct"] == 0)
    assert zero["hold_50_50_usd"] == 10
    assert zero["net_lp_usd"] == pytest.approx(9.482)
    assert zero["lp_minus_hold_usd"] == pytest.approx(-.518)
    drop = next(r for r in stress["rows"] if r["token_a_price_change_pct"] == -30)
    assert drop["hold_50_50_usd"] == pytest.approx(8.5)
    assert drop["impermanent_loss_vs_hold_pct"] < 0
    assert drop["lp_minus_hold_usd"] < 0


@pytest.mark.parametrize("bad", [0, -.5, 91, float("nan"), float("inf")])
def test_stress_invalid_shocks_rejected(bad):
    with pytest.raises(ValueError):
        v2_stress_scenarios(
            capital_usd=10, hypothetical_gross_yield_usd=.1,
            roundtrip_cost_usd=.6, shock_pct=bad
        )


def test_v3_never_gets_false_v2_concentrated_liquidity_model(tmp_path):
    payload = source(12, 600000, version="pancakeswap-amm-v3")
    r = analyze(payload, allocation=10, cost=.6, stress_pct=30)
    assert r["pools"][0]["stress_test"] is None
    root = export(r, tmp_path / "v3")
    with (root / "stress_v2.csv").open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 0
    assert "Não há teste de variação de preços para V3" in (
        root / "report.html"
    ).read_text(encoding="utf-8")


def test_reports_include_stress_csv_and_comparison_banner(tmp_path):
    r = report()
    assert r["pools"][0]["stress_test"]["rows"]
    previous = report(11, 700000)
    previous["collected_utc"] = "2026-10-02T00:00:00+00:00"
    r["collected_utc"] = "2026-10-03T00:00:00+00:00"
    r["comparison"] = compare_reports(previous, r)
    root = export(r, tmp_path / "v3")
    with (root / "stress_v2.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    html = (root / "report.html").read_text(encoding="utf-8")
    assert "cenários e dados da fonte" in html
    assert "oscilações menores" in html
    assert "Manter tokens" in html
    assert (root / "changes.json").exists()
    assert json.loads((root / "report.json").read_text(encoding="utf-8"))["stress_shock_pct"] == 30
