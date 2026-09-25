# Fase 1: contexto causal de baixo custo

A Fase 1 adiciona contexto ao connectome MaleCNS sem aumentar a quantidade de neurônios ou sinapses simuladas.

## Componentes

### Banco de features

`CausalFeatureBank` calcula retornos logarítmicos, médias, volatilidades, aceleração, persistência direcional e volume relativo em horizontes de 2, 6, 12, 24, 48 e 96 candles. A implementação usa somente a janela entregue no instante atual. Cada característica é normalizada por média e variância exponenciais causais e limitada ao intervalo `[-4, 4]`.

### Memória temporal

`DualTimescaleMemory` mantém duas médias exponenciais do sinal multiescala:

- memória rápida: `alpha=0.30`;
- memória lenta: `alpha=0.03`.

A diferença entre as duas representa aceleração ou desaceleração de contexto sem criar um novo circuito grande.

### Regimes probabilísticos

`CausalRegimeDetector` transforma tendência, persistência, volatilidade, aceleração e volume em probabilidades para `TREND_UP`, `TREND_DOWN`, `RANGE` e `SHOCK`. O regime ativo usa histerese: um novo regime precisa superar o limiar de entrada, e o regime atual é preservado enquanto sua probabilidade não cair abaixo do limiar de saída.

O estado também registra duração, regime anterior e vetor de probabilidades. Isso permite avaliar se a baixa cobertura vem do mercado, do regime ou da política de ação.

## Integração

O fluxo agora é:

```text
preços/volume
  -> CausalFeatureBank
  -> retina MaleCNS
  -> memória rápida/lenta
  -> detector probabilístico de regimes
  -> Mushroom Body e demais centros
  -> DynamicDecisionEngine
```

A memória rápida e lenta participa do sinal combinado da decisão com peso total de 20%. O detector de regimes participa do estado de decisão e mantém o bloqueio de `SHOCK`.

## Causalidade e custo

O banco não acessa outcomes futuros, não usa o split seguinte e não executa uma segunda rede profunda. O custo adicional é linear no número fixo de horizontes e características. O custo dominante continua sendo a propagação no connectome MaleCNS.

A próxima avaliação deve comparar, por split:

- distribuição dos quatro regimes;
- duração média dos regimes;
- cobertura por regime;
- acurácia, Brier Score e ECE por regime;
- memória rápida versus lenta;
- causas de `WAIT`.
