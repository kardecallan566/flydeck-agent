# Atenção temporal esparsa e sobrevivência econômica

## TemporalEventExtractor

`TemporalEventExtractor` transforma o estado causal do MaleCNS em uma assinatura compacta contendo:

```text
direção curta, média e longa
intensidade do movimento
volatilidade RMS recente
surpresa de volume
P(UP), P(DOWN), P(WAIT)
novidade
regime
```

Nenhum preço futuro entra na assinatura.

## SparseTemporalMemory

A memória mantém no máximo 512 eventos e recupera somente os 8 eventos mais relevantes por consulta. A relevância combina:

```text
similaridade cosseno do estado
intensidade do evento
decay pela idade
compatibilidade de regime
```

Cada evento recebe uma valência após o horizonte ser observado. Os readouts são separados em:

```text
up
down
flat
```

A memória aprende apenas no split de treino. Validação e teste recuperam os eventos de treino, mas não adicionam novos eventos.

## Integração com Crypto Event Policy

A política combina o sinal MaleCNS com a atenção temporal usando peso máximo de 30%. A memória não substitui a percepção atual. Eventos `flat` aumentam o risco suavemente; não existe veto absoluto.

## EconomicSurvivalMetrics

O módulo calcula sobre retornos líquidos:

```text
net_return
gross_return
volatility
max_drawdown
return_over_drawdown
sharpe_net
CVaR 95% e CVaR 99%
profit_factor
ruin_probability
recovery_periods
turnover
total_cost
average_exposure
worst_period_return
```

O retorno líquido usa custos e slippage. O turnover é calculado pela variação absoluta da posição entre sinais.

A probabilidade de ruína atual é uma métrica histórica de frequência de cruzamento da barreira de capital de 70%; ela não deve ser confundida com uma probabilidade estatística fora da amostra. A próxima extensão recomendada é bootstrap por blocos temporais.

## CLI

```bash
PYTHONPATH=src python -m flydeck.bnb_visual_cli \
  --data data/cache/BNBUSDT_5m.csv \
  --circuit data/malecns/motion_visual.json \
  --crypto-event \
  --fee-bps 5 \
  --slippage-bps 2
```

Além das métricas de posição, o CLI exibe as métricas econômicas e:

```text
temporal matches
temporal attention
```
