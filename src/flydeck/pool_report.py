"""Responsive, shareable, accessible static research page for PancakeSwap screening.

Every provider-supplied value is escaped. A positive hypothetical yield never
turns a screening result into an investment recommendation.
"""
from __future__ import annotations

import html
from datetime import datetime


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _money(n: float) -> str:
    sign = "−" if n < 0 else ""
    return sign + "US$ " + (
        f"{abs(n):,.4f}" if abs(n) < 1 else f"{abs(n):,.2f}"
    )


def _time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M UTC")
    except (ValueError, AttributeError):
        return str(iso)


def _card(pool: dict, needed: float, cost: float, *, days: int) -> str:
    esc = _escape
    v3 = pool["version"] == "V3"
    gross = float(pool["gross_yield_usd"])
    delta = float(pool["gross_minus_cost_usd"])
    apy = float(pool["reported_apy_pct"])
    if v3:
        state = "v3"
        badge, conclusion = "APY INDICATIVO", "V3: depende da faixa ativa"
    elif delta <= 0:
        state = "loss"
        badge, conclusion = "CUSTOS SUPERAM RECEITA", "Não cobre custos estimados"
    else:
        state = "positive"
        badge, conclusion = "REVISAR RISCOS", "Bruto cobre custos simulados; não é lucro"
    # Scale a marker to the ratio APY / required APR; never imply personal V3 yield.
    share = min(100, max(0, apy / needed * 100)) if needed > 0 else 100
    note = (
        "Estimativa do agregador, NÃO projeção do retorno real desta posição V3."
        if v3 else
        f"Projeção linear de {days} dias com APY de terceiros. "
        "Sem desvalorização dos tokens, perda impermanente ou impostos."
    )
    token = esc(pool["symbol"])
    warnings = "".join("<li>" + esc(w) + "</li>" for w in pool["warnings"])
    ref = esc(pool["pool_url"])
    return (
        '<article class="pool '+state+'">'
        '<div class="poolhead"><div><small class="eyebrow">'
        + esc(pool["version"]) + ' · PANCAKESWAP BSC</small>'
        '<h3>' + token + '</h3></div><span class="chip">'
        + esc(badge) + '</span></div>'
        '<div class="poolmain"><div class="earnlabel">Resultado bruto menos custo estimado</div>'
        '<div class="outcome">' + esc(_money(delta)) + '</div>'
        '<p class="interpretation">' + esc(conclusion) + '</p></div>'
        '<div class="breakdown">'
        '<div><span>Receita bruta hipotética</span><strong>' + esc(_money(gross)) + '</strong></div>'
        '<div><span>Custo total assumido</span><strong>−' + esc(_money(cost)) + '</strong></div>'
        '<div><span>APY publicado</span><strong>' + esc(f"{apy:.2f}") + '%</strong></div>'
        '<div><span>TVL informado</span><strong>' +
        esc(_money(float(pool["tvl_usd"]))) + '</strong></div>'
        '</div><div class="progressmeta"><span>APY / mínimo para custos</span>'
        '<span>'+esc(f"{apy:.1f}")+'% / '+esc(f"{needed:.1f}")+'%</span></div>'
        '<div class="track" role="img" aria-label="APY relativo ao mínimo para custos">'
        '<div class="fill" style="width:'+esc(f"{share:.1f}")+'%"></div></div>'
        '<p class="notice">' + esc(note) + '</p>'
        '<details><summary>Riscos e dados da fonte</summary><ul>'+warnings+'</ul>'
        '<p><a href="'+ref+'" target="_blank" rel="noopener noreferrer">'
        'Consultar agregador ↗</a></p>'
        '<p><a href="https://pancakeswap.finance/liquidity/pools" '
        'target="_blank" rel="noopener noreferrer">Verificar no PancakeSwap ↗</a></p>'
        '</details></article>'
    )


def render(report: dict) -> str:
    esc = _escape
    pools = report["pools"]
    v2 = [p for p in pools if p["version"] == "V2"]
    v3 = [p for p in pools if p["version"] == "V3"]
    viable_v2 = sum(p["gross_minus_cost_usd"] > 0 for p in v2)
    total = len(pools)
    needed = float(report["apy_needed_to_cover_costs_pct"])
    cost = float(report["assumed_roundtrip_cost_usd"])
    demo = report.get("synthetic_fixture_only") is True
    caution = (
        '<div class="demo" role="alert">DADOS DE DEMONSTRAÇÃO — '
        'TODAS AS POOLS E TAXAS DESTE EXEMPLO SÃO INVENTADAS.</div>'
        if demo else ""
    )
    if viable_v2 == 0:
        recommendation = (
            "Nenhuma das pools V2 exibidas cobre os custos definidos nesta simulação."
            if v2 else
            "Não há pools V2 nesta seleção; o APY agregado das V3 não é uma cotação pessoal."
        )
    else:
        recommendation = (
            f"{viable_v2} de {len(v2)} pools V2 ultrapassam apenas os custos "
            "assumidos, SEM descontar perda impermanente ou riscos de mercado."
        )
    summary = (
        '<section class="summary" aria-labelledby="decision"><div>'
        '<p class="eyebrow">DIAGNÓSTICO DO ORÇAMENTO</p>'
        '<h2 id="decision">'+esc(recommendation)+'</h2>'
        '<p>Com aporte de '+esc(_money(float(report["allocation_usd"])))+
        ', horizonte de '+esc(report["days"])+' dias e custo total estimado de '+
        esc(_money(cost))+', seriam necessários pelo menos <strong>'+
        esc(f"{needed:.1f}")+'% de APY</strong> apenas para recuperar esses custos. '
        'Esse número não inclui perdas de preço ou impostos.</p></div>'
        '<div class="decisiontag">NÃO INVESTIR TAMBÉM É UMA OPÇÃO</div></section>'
    )
    blocks = (
        ('<div class="metrics">'
         '<div><span>Orçamento</span><strong>'+esc(_money(float(report["budget_usd"])))+'</strong></div>'
         '<div><span>Aporte analisado</span><strong>'+esc(_money(float(report["allocation_usd"])))+'</strong></div>'
         '<div><span>Reserva intacta</span><strong>'+esc(_money(float(report["reserved_usd"])))+'</strong></div>'
         '<div><span>Prazo</span><strong>'+esc(report["days"])+' dias</strong></div>'
         '</div>')
    )
    groups = []
    for title, subset, identifier in (
        ("Pools V2 · análise de custos", v2, "v2"),
        ("Pools V3 · APY apenas referencial", v3, "v3")
    ):
        if subset:
            groups.append(
                '<section class="group" id="'+identifier+'">'
                '<div class="sectionheading"><h2>'+esc(title)+'</h2><span>'+
                esc(len(subset))+' resultados</span></div>'
                '<div class="cards">'+
                "".join(_card(p, needed, cost, days=report["days"]) for p in subset)+
                '</div></section>'
            )
    if not groups:
        groups = ['<section class="empty"><h2>Nenhuma pool passou nos filtros</h2>'
                  '<p>Não investir é uma decisão válida. Experimente outra faixa '
                  'de TVL para fins de pesquisa, sem reduzir a cautela.</p></section>']
    monitoring = report.get("comparison")
    changes = ""
    if monitoring:
        n = monitoring["attention_count"]
        changes = (
            '<a class="changebanner" href="changes.html"><span>'
            '<strong>Comparação com a consulta anterior</strong><br>'
            +esc(n)+' alterações merecem atenção · '+
            esc(monitoring["information_count"])+' informativas</span>'
            '<span>Consultar mudanças ↗</span></a>'
        )
    styles = StringStyles
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        '<meta name="description" content="FlyDeck: pesquisa pública de pools, '
        'custos assumidos e riscos. Não é uma recomendação de investimento.">'
        '<title>FlyDeck · Relatório de pools</title><style>'+styles+'</style>'
        '</head><body><main class="container">'
        '<header class="hero"><div class="brand"><span class="brandmark">F</span>'
        '<span>FLYDECK <em>POOL WATCH</em></span></div>'
        '<p class="eyebrow">PESQUISA DE LIQUIDEZ · FONTE PÚBLICA INDEPENDENTE</p>'
        '<h1>O rendimento compensa os custos?</h1>'
        '<p class="subtitle">Uma análise transparente para orçamentos pequenos. '
        'Os valores são hipóteses de pesquisa, não promessas de retorno.</p>'
        '<p class="date">Consultado em '+esc(_time(report["collected_utc"]))+
        ' · Dados agregados do DefiLlama</p></header>'
        +caution+summary+blocks+changes+
        '<div class="sectionheading"><h2>Comparação das pools</h2><span>'+
        esc(total)+' exibidas de '+esc(report["matching_pools"])+' filtradas</span></div>'
        '<p class="sectionnote">A ordem prioriza pares de stablecoins pelos símbolos, '
        'V2 e maior liquidez, não o maior APY. Stablecoins também podem perder paridade. '
        'Cada card detalha seus pressupostos.</p>'
        +"".join(groups)+
        '<footer><p><strong>Atenção:</strong> os rendimentos são extrapolações '
        'lineares de APYs divulgados por terceiros. Não verificamos contratos, '
        'preços reais de entrada ou saída, gás, impostos, perdas de mercado, '
        'elegibilidade em Farms nem a faixa ativa pessoal de V3.</p>'
        '<p><a href="https://defillama.com/yields" target="_blank" '
        'rel="noopener noreferrer">Fonte: DefiLlama ↗</a> · '
        '<a href="https://pancakeswap.finance/liquidity/pools" target="_blank" '
        'rel="noopener noreferrer">Verificação no PancakeSwap ↗</a></p>'
        '<p>FlyDeck · somente leitura · nenhuma carteira conectada · '
        'nenhuma transação automática</p></footer>'
        '</main></body></html>'
    )


StringStyles = r"""
:root{color-scheme:dark;font-family:Inter,Segoe UI,system-ui,-apple-system,sans-serif;
--bg:#0c1421;--surface:#17293a;--surface2:#1b3043;--line:#365164;
--text:#f0f7fa;--muted:#afc6d3;--mint:#8de1d4;--warn:#ffca95;--negative:#ffc1bd}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font-size:15px;line-height:1.55}.container{max-width:1350px;margin:auto;padding:30px 26px 50px}
h1,h2,h3,p{overflow-wrap:anywhere}.hero{padding:26px 0 28px}
.brand{display:flex;align-items:center;gap:10px;font-weight:850;letter-spacing:.16em;
font-size:13px}.brand em{font-style:normal;color:var(--mint);margin-left:5px}
.brandmark{background:var(--mint);color:#0a2028;border-radius:11px;
width:35px;height:35px;display:grid;place-items:center;font-size:19px;letter-spacing:0}
.eyebrow{color:var(--mint);font-weight:700;font-size:12px;letter-spacing:.11em}
.hero>.eyebrow{margin:28px 0 6px}h1{font-size:clamp(29px,4vw,46px);
letter-spacing:-.045em;line-height:1.12;margin:0}
.subtitle{font-size:16px;color:#c8d9e2;max-width:790px;margin:14px 0}
.date,.sectionnote,footer{color:var(--muted)}.date{font-size:12px;margin-top:20px}
.demo{margin:0 0 18px;background:#5a3a25;color:#fff1d9;
border:1px solid #f2be72;border-radius:13px;padding:15px;font-weight:bold}
.summary{border:1px solid #547a88;background:linear-gradient(115deg,#1b3747,#15283a);
border-radius:18px;padding:25px 28px;display:flex;align-items:center;justify-content:space-between;gap:25px}
.summary h2{font-size:clamp(19px,2.4vw,26px);letter-spacing:-.025em;line-height:1.25;margin:5px 0 12px}
.summary p:last-child{max-width:850px;color:#c7d8e0;margin:0}
.decisiontag{font-size:11px;letter-spacing:.06em;color:#ffe3b8;background:#5c4430;
border:1px solid #91714b;padding:10px 14px;text-align:center;border-radius:9px;
font-weight:800;max-width:230px;flex-shrink:0}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 34px}
.metrics>div{border:1px solid var(--line);border-radius:14px;padding:17px;background:var(--surface)}
.metrics span{display:block;font-size:12px;color:var(--muted)}
.metrics strong{display:block;font-size:clamp(18px,2vw,25px);margin-top:5px;font-variant-numeric:tabular-nums}
.changebanner{display:flex;justify-content:space-between;align-items:center;gap:14px;
padding:20px;background:#203d44;border:1px solid #5d988d;border-radius:14px;
color:#ebfff8;text-decoration:none;margin-bottom:30px}
.changebanner span:last-child{color:var(--mint);font-weight:700;white-space:nowrap}
.sectionheading{display:flex;justify-content:space-between;align-items:baseline;gap:15px;margin:32px 0 10px}
.sectionheading h2{font-size:clamp(20px,2vw,26px);margin:0;letter-spacing:-.02em}
.sectionheading>span{color:var(--muted);font-size:12px}
.sectionnote{margin:0 0 22px;max-width:900px}
.group{margin:24px 0 38px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:17px}
.pool,.empty{border:1px solid var(--line);border-radius:16px;background:var(--surface);
min-width:0;padding:21px;display:flex;flex-direction:column}
.poolhead{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;min-height:86px}
.poolhead h3{margin:6px 0;font-size:clamp(18px,1.8vw,23px);letter-spacing:-.025em}
.poolhead small{font-size:10px}.chip{font-size:10px;font-weight:800;letter-spacing:.02em;
padding:6px 9px;border-radius:8px;max-width:133px;text-align:center}
.loss .chip{background:#503a39;color:#ffd1cc}.positive .chip{background:#24483f;color:#b4fae0}
.v3 .chip{background:#4c4432;color:#ffe2aa}
.poolmain{border-top:1px solid #355063;padding:16px 0 11px}
.earnlabel{color:var(--muted);font-size:12px}.outcome{font-size:clamp(24px,2.6vw,33px);
font-variant-numeric:tabular-nums;font-weight:850;letter-spacing:-.025em;margin:4px 0}
.loss .outcome{color:var(--negative)}.v3 .outcome{color:var(--warn)}
.positive .outcome{color:#aff6d7}.interpretation{font-size:12px;color:var(--muted);margin:2px 0 0}
.breakdown{border-top:1px solid #355063;padding:12px 0;display:grid;gap:8px}
.breakdown>div{display:flex;justify-content:space-between;align-items:baseline;gap:10px;font-size:12px}
.breakdown span{color:var(--muted)}.breakdown strong{font-variant-numeric:tabular-nums;text-align:right}
.progressmeta{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:11px}
.track{height:6px;background:#31495a;border-radius:20px;overflow:hidden;margin:8px 0 15px}
.fill{height:100%;border-radius:20px;background:var(--mint)}
.notice{font-size:12px;color:#d2dee3;margin:0 0 15px;line-height:1.6}
details{margin-top:auto;border-top:1px solid #355063;padding-top:13px}
summary{cursor:pointer;color:var(--mint);font-size:13px;font-weight:700}
details li{color:#c6d5dd;margin:9px 0;font-size:12px}
details p{font-size:12px}a{color:var(--mint)}a:hover{text-decoration:none}
.empty{padding:28px}footer{border-top:1px solid #304a5e;padding-top:26px;font-size:12px;line-height:1.75}
footer p{max-width:950px}
@media(max-width:1050px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:690px){.container{padding:20px 14px}.summary{display:block;padding:21px}
.decisiontag{margin-top:18px;max-width:none}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
.cards{grid-template-columns:minmax(0,1fr)}.sectionheading{display:block}
.changebanner{display:block}.changebanner span:last-child{display:block;margin-top:8px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
