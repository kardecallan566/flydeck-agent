"""Deterministic, offline comparison of two public pool research snapshots.

No background monitoring or wallet access. New/absent means passing/not passing
the *source+filters*, never newly deployed or permanently removed on chain.
"""
from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime, timezone
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
    try:
        previous_time = datetime.fromisoformat(
            str(previous["collected_utc"]).replace("Z", "+00:00")
        )
        current_time = datetime.fromisoformat(
            str(current["collected_utc"]).replace("Z", "+00:00")
        )
        if previous_time.tzinfo is None or current_time.tzinfo is None:
            raise ValueError("timestamps require explicit UTC timezone")
        minutes = (current_time - previous_time).total_seconds() / 60
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp in comparison reports") from exc
    if minutes <= 0:
        raise ValueError("current report must be strictly later than previous")
    prior, prior_full = _index(previous)
    recent, recent_full = _index(current)
    events = []
    movements = []
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
        movements.append({
            **common,
            "abs_apy_change_pp": abs(apy_delta),
            "abs_tvl_change_pct": (
                round(abs(tvl_change), 3) if tvl_change is not None else None
            ),
            "triggered_apy_threshold": abs(apy_delta) >= apy_change_pp,
            "triggered_tvl_threshold": (
                tvl_change is not None and tvl_change <= -tvl_drop_pct
            ),
        })
        if apy_delta <= -apy_change_pp:
            events.append({**common, "event": "APY_DROP",
                           "severity": "attention"})
        elif apy_delta >= apy_change_pp:
            events.append({**common, "event": "APY_RISE_UNVERIFIED",
                           "severity": "information"})
        if tvl_change is not None and tvl_change <= -tvl_drop_pct:
            events.append({**common, "event": "TVL_DROP",
                           "severity": "attention"})
    # Include small observed changes even when none crosses the alert limits.
    # DO NOT extrapolate a minute of data into a daily yield trend.
    movements.sort(key=lambda p: (
        -p["abs_apy_change_pp"],
        -(p["abs_tvl_change_pct"] or 0),
        p["symbol"], p["pool_id"],
    ))
    same_filters = previous.get("min_tvl_usd") == current.get("min_tvl_usd")
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
        "interval_minutes": round(minutes, 3),
        "short_observation_window": minutes < 60,
        "matching_pools": len(prior.keys() & recent.keys()),
        "measured_movements": len(movements),
        "unchanged_to_precision": sum(
            m["apy_change_pp"] == 0 and m["tvl_change_pct"] == 0
            for m in movements
        ),
        "minor_changes": sum(
            (m["apy_change_pp"] != 0 or m["tvl_change_pct"] != 0)
            and not m["triggered_apy_threshold"]
            and not m["triggered_tvl_threshold"]
            for m in movements
        ),
        "top_movements": movements[:8],
        "filters_comparable": same_filters,
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
    """Cost-aware, compact public-feed comparison; never labels changes as profit."""
    esc = lambda value: html.escape(str(value), quote=True)
    names = {
        "APY_DROP": "Queda no APY divulgado",
        "APY_RISE_UNVERIFIED": "Aumento no APY divulgado — verificar",
        "TVL_DROP": "Queda na liquidez publicada",
        "ENTERED_SOURCE_FILTER": "Entrou na seleção do agregador",
        "LEFT_SOURCE_FILTER": "Saiu dos filtros ou do agregador",
    }

    def date(iso: str) -> str:
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime(
                "%d/%m/%Y às %H:%M:%S UTC"
            )
        except (ValueError, AttributeError):
            return str(iso)

    def number(value, *, suffix="", signed=False) -> str:
        if value is None:
            return "indisponível"
        n = float(value)
        sign = "+" if signed and n > 0 else ""
        return f"{sign}{n:,.3f}{suffix}"

    def money(value) -> str:
        return "US$ " + number(value)

    event_cards = []
    for event in changes["events"]:
        detail = []
        if "apy_change_pp" in event:
            detail.append("APY " + esc(number(
                event["apy_change_pp"], suffix=" p.p.", signed=True
            )))
        if event.get("tvl_change_pct") is not None:
            detail.append("TVL " + esc(number(
                event["tvl_change_pct"], suffix="%", signed=True
            )))
        if "new_apy_pct" in event and "old_apy_pct" in event:
            detail.append(
                "APY anterior " + esc(number(event["old_apy_pct"], suffix="%"))
                + " → " + esc(number(event["new_apy_pct"], suffix="%"))
            )
        event_cards.append(
            '<article class="event ' + esc(event["severity"]) + '">'
            '<span class="eyebrow">' + esc(names[event["event"]]) + '</span>'
            '<h3>' + esc(event["symbol"]) + '</h3><p>'
            + " · ".join(detail) + '</p><small>Identificador: '
            + esc(event["pool_id"]) + '</small></article>'
        )

    movement_rows = []
    for row in changes.get("top_movements", []):
        movement_rows.append(
            '<tr><th scope="row">' + esc(row["symbol"]) + '</th>'
            '<td>' + esc(number(row["old_apy_pct"], suffix="%")) + '</td>'
            '<td>' + esc(number(row["new_apy_pct"], suffix="%")) + '</td>'
            '<td class="num">' + esc(number(
                row["apy_change_pp"], suffix=" p.p.", signed=True
            )) + '</td><td class="num">' + esc(number(
                row.get("tvl_change_pct"), suffix="%", signed=True
            )) + '</td></tr>'
        )

    elapsed = changes.get("interval_minutes")
    if elapsed is None:
        elapsed_text = "Intervalo indisponível (relatório antigo)"
    elif elapsed < 60:
        elapsed_text = f"{elapsed:.1f} minutos"
    else:
        elapsed_text = f"{elapsed / 60:.1f} horas"
    if changes.get("short_observation_window"):
        short_note = (
            '<p class="notice">As consultas foram realizadas com menos de uma '
            'hora de diferença. Dados de agregadores podem ser atualizados em '
            'intervalos maiores: ausência de alerta NÃO confirma estabilidade '
            'nem significa que a fonte divulgou dados novos.</p>'
        )
    else:
        short_note = ""

    coverage = (
        "Comparação ampla: ambas as consultas guardaram o índice completo "
        "das pools que passaram nos mesmos filtros."
        if changes.get("comparable_full_universe") else
        "Comparação parcial: uma consulta pode conter somente os cards "
        "exibidos ou filtros de TVL diferentes. Entradas/saídas da lista "
        "não comprovam mudanças nos contratos."
    )
    events = "".join(event_cards) if event_cards else (
        '<div class="empty"><h3>Nenhum alerta atingiu os limites escolhidos.</h3>'
        '<p>As menores oscilações observadas aparecem na tabela abaixo. '
        'Isso não demonstra estabilidade do mercado.</p></div>'
    )
    movement_table = (
        '<div class="scroll"><table><thead><tr><th>Pool</th><th>APY antes</th>'
        '<th>APY agora</th><th>Variação APY</th><th>Variação TVL</th>'
        '</tr></thead><tbody>' + "".join(movement_rows) + '</tbody></table></div>'
        if movement_rows else
        "<p>Não existem pools em comum com valores válidos para comparar.</p>"
    )
    style = """
    :root{color-scheme:dark;font:15px system-ui,Segoe UI,Arial;
    background:#0c1421;color:#edf7f9}*{box-sizing:border-box}
    body{margin:0;line-height:1.55}.container{max-width:1140px;margin:auto;
    padding:27px 22px 55px}.eyebrow{color:#9ddbd5;font-size:12px;
    letter-spacing:.1em;font-weight:750}h1{font-size:clamp(27px,4vw,44px);
    letter-spacing:-.04em;margin:8px 0}.muted{color:#b7cad6}
    .hero{margin-bottom:27px}.summary{display:grid;grid-template-columns:
    repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:25px 0}
    .metric,.event,.empty,.notice{border:1px solid #385063;
    border-radius:14px;background:#172b3b;padding:17px}
    .metric span{display:block;color:#b7cad6;font-size:12px}
    .metric strong{font-size:25px;display:block;margin-top:5px}
    .notice{border-color:#b08751;color:#f6d6a6;background:#312c2a}
    .details{color:#b7cad6;font-size:13px}
    .events{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:12px}.event{min-width:0}.event.attention{border-left:5px solid #e9aa70}
    .event.information{border-left:5px solid #86d6cc}.event h3{font-size:20px;
    margin:8px 0}.event p{margin:0 0 12px}.event small{color:#9fb7c6}
    section{margin:30px 0}h2{font-size:22px}.scroll{overflow-x:auto;
    border:1px solid #385063;border-radius:12px}
    table{border-collapse:collapse;width:100%;min-width:610px}
    th,td{padding:12px 14px;border-bottom:1px solid #304657;text-align:right;
    font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}
    thead{color:#bad3de;background:#223749}tbody tr:last-child>*{border:0}
    a{color:#93e0d6}footer{font-size:12px;color:#b7cad6;
    border-top:1px solid #354858;padding-top:20px}
    """
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        '<title>FlyDeck · Comparação entre consultas</title>'
        '<style>' + style + '</style></head><body><main class="container">'
        '<header class="hero"><p class="eyebrow">FLYDECK · MONITORAMENTO PÚBLICO'
        '</p><h1>O que mudou desde a última consulta?</h1>'
        '<p class="muted">Anterior: ' + esc(date(changes["prior_utc"])) +
        '<br>Atual: ' + esc(date(changes["current_utc"])) +
        '<br>Intervalo: ' + esc(elapsed_text) + '</p></header>'
        '<section class="summary">'
        '<div class="metric"><span>Alterações que merecem atenção</span><strong>'
        + esc(changes["attention_count"]) + '</strong></div>'
        '<div class="metric"><span>Alterações informativas</span><strong>'
        + esc(changes["information_count"]) + '</strong></div>'
        '<div class="metric"><span>Pools nas duas consultas</span><strong>'
        + esc(changes.get("matching_pools", "—")) + '</strong></div>'
        '<div class="metric"><span>Mudanças menores que os limites</span><strong>'
        + esc(changes.get("minor_changes", "—")) + '</strong></div></section>'
        + short_note +
        '<p class="details">' + esc(coverage) + '</p>'
        '<section><h2>Alertas configurados</h2>'
        '<p class="muted">Limites: APY ±'
        + esc(changes["thresholds"]["apy_delta_percentage_points"])
        + ' p.p. ou queda de TVL ≥'
        + esc(changes["thresholds"]["tvl_drop_pct"])
        + '%. APY maior não significa rendimento seguro.</p>'
        '<div class="events">' + events + '</div></section>'
        '<section><h2>Maiores variações observadas</h2>'
        '<p class="muted">Até oito pools em comum, por mudança absoluta '
        'do APY publicado — inclusive quando nenhuma atingiu o limiar '
        'de alerta. Não extrapolar para tendências diárias.</p>'
        + movement_table + '</section>'
        '<footer><p>Fonte: dados públicos agregados; não são medições '
        'diretas nos contratos. Uma pool ausente pode simplesmente estar '
        'fora do filtro TVL ou do resultado fornecido.</p>'
        '<p>' + esc(changes["disclaimer"]) + '</p></footer>'
        '</main></body></html>'
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
