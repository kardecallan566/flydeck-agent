"""Deterministic, offline comparison of two public pool research snapshots.

No background monitoring or wallet access. New/absent means passing/not passing
the *source+filters*, never newly deployed or permanently removed on chain.
"""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


def _index(report: dict) -> tuple[dict[str, dict], bool]:
    if not isinstance(report, dict) or report.get("research_only") is not True:
        raise ValueError("not a supported Pool Watch research report")
    full = report.get("monitoring_index")
    complete = isinstance(full, list)
    entries = full if complete else report.get("pools")
    if not isinstance(entries, list):
        raise ValueError("previous/current report lacks a pool index")
    result = {}
    for p in entries:
        if not isinstance(p, dict):
            continue
        key = p.get("pool_id")
        if not isinstance(key, str) or not key or key in result:
            continue
        try:
            apy = float(p["reported_apy_pct"])
            tvl = float(p["tvl_usd"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (apy, tvl))) or tvl < 0:
            continue
        result[key] = p
    return result, complete


def compare_reports(previous: dict, current: dict, *,
                    apy_change_pp: float = 2.0,
                    tvl_drop_pct: float = 20.0) -> dict:
    """Compare source-filtered pools by stable aggregator ID, NOT symbol."""
    if (not 0 < apy_change_pp <= 10000
        or not 0 < tvl_drop_pct <= 100
        or not all(map(math.isfinite, (apy_change_pp, tvl_drop_pct)))):
        raise ValueError("thresholds must be finite and strictly positive")
    if previous.get("source") != current.get("source"):
        raise ValueError("reports use different feeds; comparison unavailable")
    if previous.get("synthetic_fixture_only") != current.get("synthetic_fixture_only"):
        raise ValueError("cannot compare synthetic and real feed reports")
    if previous.get("collected_utc") == current.get("collected_utc"):
        raise ValueError("reports have equal collection timestamps")
    prior, prior_full = _index(previous)
    recent, recent_full = _index(current)
    events = []
    for key in sorted(prior.keys() & recent.keys()):
        a, b = prior[key], recent[key]
        apy_delta = round(
            float(b["reported_apy_pct"]) - float(a["reported_apy_pct"]), 5
        )
        a_tvl = float(a["tvl_usd"])
        tvl_change = (float(b["tvl_usd"]) / a_tvl - 1) * 100 if a_tvl else None
        symbol = str(b["symbol"])[:80]
        common = {"pool_id": key, "symbol": symbol,
                  "old_apy_pct": a["reported_apy_pct"],
                  "new_apy_pct": b["reported_apy_pct"],
                  "apy_change_pp": apy_delta,
                  "old_tvl_usd": a_tvl,
                  "new_tvl_usd": b["tvl_usd"],
                  "tvl_change_pct": round(tvl_change, 3)
                  if tvl_change is not None else None}
        if apy_delta <= -apy_change_pp:
            events.append({**common, "event": "APY_DROP",
                           "severity": "attention"})
        elif apy_delta >= apy_change_pp:
            events.append({**common, "event": "APY_RISE_UNVERIFIED",
                           "severity": "information"})
        if tvl_change is not None and tvl_change <= -tvl_drop_pct:
            events.append({**common, "event": "TVL_DROP",
                           "severity": "attention"})
    for key in sorted(recent.keys() - prior.keys()):
        p = recent[key]
        events.append({"pool_id": key, "symbol": str(p["symbol"])[:80],
                       "event": "ENTERED_SOURCE_FILTER",
                       "severity": "information",
                       "new_apy_pct": p["reported_apy_pct"],
                       "new_tvl_usd": p["tvl_usd"]})
    for key in sorted(prior.keys() - recent.keys()):
        p = prior[key]
        events.append({"pool_id": key, "symbol": str(p["symbol"])[:80],
                       "event": "LEFT_SOURCE_FILTER",
                       "severity": "attention",
                       "old_apy_pct": p["reported_apy_pct"],
                       "old_tvl_usd": p["tvl_usd"]})
    events.sort(key=lambda e: (
        e["severity"] != "attention", e["event"], e["symbol"], e["pool_id"]
    ))
    return {
        "schema_version": 1, "research_only": True,
        "comparison_is_public_feed_not_onchain": True,
        "prior_utc": previous["collected_utc"],
        "current_utc": current["collected_utc"],
        "prior_complete_index": prior_full,
        "current_complete_index": recent_full,
        "comparable_full_universe": prior_full and recent_full
        and previous.get("min_tvl_usd") == current.get("min_tvl_usd"),
        "thresholds": {"apy_delta_percentage_points": apy_change_pp,
                       "tvl_drop_pct": tvl_drop_pct},
        "old_pool_count": len(prior),
        "new_pool_count": len(recent),
        "attention_count": sum(e["severity"] == "attention" for e in events),
        "information_count": sum(e["severity"] == "information" for e in events),
        "events": events,
        "disclaimer": (
            "An absent pool may simply fail the TVL filter, be missing from "
            "the source or be outside an old top-N snapshot; never assume "
            "on-chain removal. APY changes are not guaranteed profits. "
            "Cross-report comparison of different min-TVL filters is incomplete."
        ),
    }


def render_alerts_html(changes: dict) -> str:
    esc = lambda v: html.escape(str(v), quote=True)
    titles = {
        "APY_DROP": "Queda no APY divulgado",
        "APY_RISE_UNVERIFIED": "Aumento no APY divulgado — conferir",
        "TVL_DROP": "Redução de liquidez informada",
        "ENTERED_SOURCE_FILTER": "Passou nos filtros atuais",
        "LEFT_SOURCE_FILTER": "Saiu dos filtros ou dos dados",
    }
    lines = []
    for e in changes["events"]:
        details = []
        if "apy_change_pp" in e:
            details.append("Δ APY: " + esc(e["apy_change_pp"]) + " p.p.")
        if "tvl_change_pct" in e and e["tvl_change_pct"] is not None:
            details.append("Δ TVL: " + esc(e["tvl_change_pct"]) + "%")
        lines.append(
            '<article class="event ' + esc(e["severity"]) + '">'
            '<span class="eyebrow">' + esc(titles[e["event"]]) + '</span>'
            '<h2>' + esc(e["symbol"]) + '</h2><p>' +
            " · ".join(details) +
            '</p><small>Referência agregada: ' + esc(e["pool_id"]) + '</small>'
            '</article>'
        )
    body = "".join(lines) if lines else (
        "<p>Nenhuma variação atingiu os limites definidos nas pools "
        "observadas. Isso não confirma que o mercado esteja estável.</p>"
    )
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>FlyDeck · Mudanças observadas</title>'
        '<style>body{font:16px system-ui;background:#0c1422;color:#edf5f8;'
        'max-width:900px;margin:auto;padding:28px;line-height:1.55}'
        '.event{border:1px solid #354a5d;border-radius:15px;padding:20px;'
        'background:#172a3b;margin:14px 0}.attention{border-left:5px solid #e6aa68}'
        '.information{border-left:5px solid #6dbbb6}'
        '.eyebrow,small{color:#b9cbd8}h1{font-size:2rem}h2{margin:.3em 0}'
        '.caution{padding:18px;border:1px solid #94794a;border-radius:12px}'
        'a{color:#8be4e0}</style></head><body>'
        '<p>FLYDECK · COMPARAÇÃO DE FONTES PÚBLICAS</p>'
        '<h1>O que mudou desde a última consulta?</h1>'
        '<p>Anterior: ' + esc(changes["prior_utc"]) +
        '<br>Atual: ' + esc(changes["current_utc"]) + '</p>'
        '<p class="caution">' + esc(changes["disclaimer"]) + '</p>'
        + body + '</body></html>'
    )


def save_comparison(changes: dict, root: Path) -> None:
    root = Path(root)
    (root / "changes.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "changes.html").write_text(
        render_alerts_html(changes), encoding="utf-8",
    )
    fields = ("symbol", "event", "severity", "old_apy_pct", "new_apy_pct",
              "apy_change_pp", "old_tvl_usd", "new_tvl_usd",
              "tvl_change_pct", "pool_id")
    with (root / "changes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in changes["events"]:
            writer.writerow({field: event.get(field) for field in fields})
