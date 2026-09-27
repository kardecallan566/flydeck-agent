# Protocolo de ablação

A matriz compara as mesmas divisões cronológicas, dataset, circuito e custos:

```text
baseline_original
malecns_risk
current_temporal_crypto_event
```

## Definições únicas

- **accuracy**: decisões não-WAIT corretas divididas por decisões não-WAIT.
- **hit rate**: decisões com retorno líquido positivo depois de custo divididas por entradas.
- **coverage**: entradas divididas pelo número total de rounds.
- **economic return**: composição dos retornos líquidos candle a candle.

Accuracy mede direção. Hit rate mede resultado econômico por entrada. Elas não devem ser comparadas como se fossem a mesma métrica.

## Execução

```powershell
python -m flydeck.bnb_visual_cli `
  --data D:\flydeck-agent\data\cache\BNBUSDT_5m.csv `
  --circuit D:\flydeck-agent\data\malecns\motion_visual.json `
  --ablation `
  --fee-bps 5 `
  --slippage-bps 2
```

O comando imprime uma linha por variante e split com:

```text
accuracy
hit_rate
coverage
economic_return
profit_factor
max_drawdown
```

## Atenção temporal

A memória usa um piso de peso de longo prazo e um limiar de recuperação menor. Isso corrige o caso em que o gap entre o fim do treino e o teste fazia todos os scores decaírem abaixo de zero. A memória continua congelada em validação e teste; esses splits não adicionam eventos.
