# Fase 2: política probabilística e abstinência calibrada

A Fase 2 separa a leitura de evidência do connectome da escolha final de `UP`, `DOWN` ou `WAIT`. A nova cabeça é pequena e recebe somente três scores compactos, a incerteza e a novidade. Ela não duplica o MaleCNS nem propaga atividade por outro connectome.

## Cabeça de ação

`ActionProbabilityHead` transforma a evidência em três probabilidades:

```text
P(WAIT), P(UP), P(DOWN)
```

A transformação usa softmax com temperatura. A temperatura inicial é conservadora e pode ser ajustada durante o treino por uma atualização causal baseada no outcome do candle seguinte. O outcome observado em `t+1` atualiza a predição registrada em `t`; ele nunca influencia a decisão já tomada em `t`.

Durante validação e teste, a cabeça fica congelada. A temperatura aprendida no treino é preservada entre os splits, enquanto os estados transitórios são resetados.

## Política de abstinência

A política não entra apenas porque um score bruto é maior que o outro. Ela exige simultaneamente:

- probabilidade direcional mínima;
- margem mínima entre `P(UP)` e `P(DOWN)`;
- ausência de conflito ou incerteza excessivos;
- ausência de choque;
- limiar de confiança compatível com o estado metabólico.

`WAIT` continua obrigatório em `SHOCK`. Em estados normais, `WAIT` é escolhido quando a probabilidade de abstinência domina ou a margem direcional é estreita.

## Métricas multiclass

O benchmark agora calcula Brier Score e ECE sobre todas as três classes, incluindo as previsões `WAIT`.

Para cada candle:

```text
Brier_t = (P(WAIT)-Y(WAIT))²
         +(P(UP)-Y(UP))²
         +(P(DOWN)-Y(DOWN))²
```

O Brier final é a média sobre todos os candles do split, e não somente sobre as entradas. O ECE usa a confiança máxima da distribuição de três classes e verifica se a classe de maior probabilidade corresponde ao outcome observado.

As métricas direcionais por entrada continuam disponíveis nas métricas por regime. Assim, é possível distinguir calibração global, qualidade das entradas e comportamento de abstinência.

## Leitura esperada

Um sistema saudável deve ser avaliado em conjunto:

```text
accuracy direcional
coverage
P(WAIT) médio
Brier multiclass
ECE multiclass
Brier/ECE por regime
```

Reduzir o ECE sem aumentar artificialmente a cobertura é preferível a forçar entradas. Por outro lado, Brier baixo com `WAIT` em quase todos os candles pode significar que o modelo aprendeu somente a abstinência; por isso a cobertura continua obrigatória no relatório.
