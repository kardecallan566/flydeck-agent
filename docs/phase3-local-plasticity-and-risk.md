# Fase 3: plasticidade local, memória episódica e política de risco

A Fase 3 adiciona aprendizagem e controle sem atualizar todas as sinapses do connectome MaleCNS em cada candle.

## Traços de elegibilidade

`SparseEligibilityTrace` mantém somente características ativas em um dicionário esparso. A cada candle os traços sofrem decaimento; quando chega o retorno causal do candle seguinte, o reward multiplica os traços e produz atualizações limitadas.

O módulo está separado da propagação dos 30 mil neurônios. Ele pode ser conectado a readouts locais e à cabeça de ação sem modificar todo o grafo. Os limites de atualização evitam que um único candle extremo domine a aprendizagem.

## Memória episódica

`BoundedEpisodicMemory` armazena no máximo 256 vetores compactos, com ação, reward e regime. A memória é FIFO e pode ser congelada durante validação e teste. A distância ao episódio mais próximo produz uma medida de novidade:

```text
novidade baixa  -> estado parecido com o treino
novidade alta   -> estado pouco conhecido
```

A novidade participa da política de abstinência. Estados muito desconhecidos podem gerar `WAIT` sem alterar os pesos do MaleCNS.

## Política de risco

`LightweightRiskPolicy` é determinística e fica fora do cérebro neural. Ela controla:

- perdas consecutivas;
- cooldown após sequência de perdas;
- drawdown acumulado;
- novidade extrema.

A política pode transformar uma entrada em `WAIT_RISK_POLICY`, mas não altera sinapses do connectome. Isso separa previsão de proteção de capital.

## Aprendizado causal

A sequência é:

```text
estado em t
  -> decisão e ação
  -> candle seguinte observado
  -> reward causal
  -> elegibilidade e cabeça de ação atualizadas
  -> memória episódica recebe o episódio
```

Durante validação e teste, a cabeça probabilística e a memória episódica são congeladas. O estado transitório é reinicializado entre splits.

## Limites

Esta fase não garante maior acurácia. Ela torna o aprendizado mais localizado e a avaliação mais interpretável. O resultado deve ser analisado por cobertura, perdas consecutivas, drawdown, novidade, causas de `WAIT` e métricas de calibração.
