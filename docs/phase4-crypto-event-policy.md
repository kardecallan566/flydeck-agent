# Fase 4: Crypto Event Policy

A Fase 4 separa a representação visual do MaleCNS da decisão econômica. O connectome continua produzindo sinais de movimento, contexto, memória e incerteza; a nova política transforma esses sinais em uma exposição contínua.

## Saída da política

Cada candle produz:

```text
position:        -1.0 até +1.0
direction:       -1.0 até +1.0
confidence:      0.0 até 1.0
horizon:         1, 3 ou 6 candles
expected_return
risk
```

Valores positivos representam exposição comprada e valores negativos representam exposição vendida. Uma posição próxima de zero significa exposição pequena, mas não depende de uma regra binária `WAIT`.

O sizing padrão é limitado a `35%` de exposição. A política exige edge direcional mínimo de `0.18` e não considera `P(WAIT)` como confiança direcional. Se a cabeça estiver em abstinência, a posição é reduzida suavemente em vez de receber exposição alta por engano.

## Rótulos econômicos

`CryptoEventLabeler` calcula retornos futuros em três horizontes:

```text
1 candle
3 candles
6 candles
```

O limiar de cada horizonte inclui:

```text
custos de ida e volta
margem mínima
volatilidade local ajustada pela raiz do horizonte
```

O rótulo é:

```text
UP:
    retorno > limiar líquido

DOWN:
    retorno < -limiar líquido

WAIT:
    movimento não paga os custos e a margem
```

Os valores futuros são usados somente no benchmark para medir o resultado da decisão já tomada. Eles não entram nas features ou na política em tempo de decisão.

## PnL e risco

O benchmark walk-forward calcula:

```text
PnL líquido
máximo drawdown
volatilidade dos retornos por sinal
Sharpe-like anualizado para candles de 5 minutos
hit rate líquido
posição média
horizonte médio
```

O PnL de cada sinal é:

```text
position × retorno realizado - |position| × custos de ida e volta
```

Os custos padrão são:

```text
fee:      5 bps por lado
slippage: 2 bps por lado
```

Eles podem ser alterados no CLI.

## Regime e novidade

Regimes e novidade são fatores contínuos de risco. `SHOCK` reduz a exposição, mas não bloqueia automaticamente uma oportunidade. Novidade também reduz tamanho de posição gradualmente em vez de converter todo estado novo em `WAIT`.

## Execução

```bash
PYTHONPATH=src python -m flydeck.bnb_visual_cli \
  --data data/cache/BNBUSDT_5m.csv \
  --circuit data/malecns/motion_visual.json \
  --crypto-event \
  --fee-bps 5 \
  --slippage-bps 2
```

O benchmark usa a divisão cronológica:

```text
70% treino
15% validação congelada
15% teste congelado
```

## Critério de aprovação

A política não deve ser aprovada por acurácia direcional isolada. A avaliação deve comparar contra uma política simples e observar:

```text
retorno líquido positivo
controle de drawdown
estabilidade entre validação e teste
cobertura/exposição não nula
custos e slippage incluídos
```
