# Diagnóstico e plano de aprimoramento do FlyDeck Agent

## Conclusão

Os resultados com 76 mil candles não justificam continuar aumentando o número de rodadas da versão anterior. A acurácia permaneceu próxima de 50% e a validação/teste apresentaram forte concentração em DOWN. O problema é compatível com um deslocamento de distribuição entre períodos e com um sinal direcional não calibrado, não com falta simples de dados.

A primeira correção aplicada nesta versão é uma **calibração online simétrica**. O motor mantém centros exponenciais separados para a evidência sensorial e para a diferença entre `Q(UP)` e `Q(DOWN)`. A decisão usa o sinal centrado, em vez do valor absoluto acumulado. Essa mudança reduz a possibilidade de um nível médio negativo transformar todo o período de validação em DOWN, sem usar o candle futuro.

## Evidência metodológica

A avaliação deve preservar a ordem temporal e repetir o teste em múltiplas janelas. Um estudo recente de walk-forward destaca a disciplina do conjunto de informação, a validação rolling e os custos realistas como salvaguardas contra look-ahead e overfitting [1]. Outro trabalho sobre avaliação de backtests encontrou vantagens de métodos purged e combinatórios em ambientes sintéticos, mas a conclusão não elimina a necessidade de walk-forward quando o objetivo é simular implantação temporal [2].

A detecção de regimes também é relevante. Um framework regime-aware recente usa um modelo online de estados de mercado, reestimado somente com dados passados, além de janelas walk-forward e métricas de calibração como Brier Score e Expected Calibration Error [3]. Para o FlyDeck, a versão inicial pode usar um detector causal mais simples e auditável, baseado em tendência, volatilidade e persistência, antes de introduzir HMM.

Confiança não deve ser tratada como probabilidade sem calibração. Curvas de calibração comparam a confiança prevista com a frequência observada; Brier Score e log loss avaliam conjuntamente calibração, resolução e incerteza. A documentação do scikit-learn recomenda separar os dados usados para ajustar o classificador dos dados usados para ajustar o calibrador [4].

## Mudanças aplicadas

O `DynamicDecisionEngine` agora mantém dois centros online:

```text
centro_evidencia(t) = EMA(combined_signal)
centro_MBON(t)      = EMA(Q(UP) - Q(DOWN))
```

A direção usada para decidir é:

```text
direcao = 0,70 × (evidencia - centro_evidencia)
         + 0,30 × ((Q(UP) - Q(DOWN)) - centro_MBON)
```

A atualização dos centros usa somente o estado atual, antes de qualquer resultado futuro. O `reset()` zera esses centros entre treino, validação e teste. Os pesos contextuais do Mushroom Body continuam persistentes durante o treino e congelados na avaliação.

## Próximas melhorias priorizadas

A próxima etapa deve adicionar ao relatório de cada split a média, o desvio e os quantis de `combined_signal`, `Q(UP)`, `Q(DOWN)`, `Q(WAIT)` e da distribuição real dos outcomes. Isso permitirá distinguir um viés do circuito de um viés da distribuição de mercado.

Depois, deve ser implementado um detector causal de quatro regimes: tendência de alta, tendência de baixa, consolidação e choque. O detector deve usar somente retornos passados, volatilidade rolling, persistência de sinal e amplitude relativa. A política pode então exigir margem maior em consolidação e bloquear entradas em choque.

A avaliação deve evoluir de um único split para walk-forward com várias janelas. Cada janela deve ajustar pesos e qualquer calibrador apenas no passado, aplicar uma pequena purga correspondente ao horizonte de previsão e avaliar no bloco seguinte. O resultado final deve incluir média, mediana e intervalo entre janelas, além de acurácia por regime.

Por fim, a meta de 60%–70% deve ser tratada como hipótese a ser testada, não como garantia. A aceitação mínima deve exigir acurácia fora da amostra acima do baseline, presença de UP e DOWN nos splits, estabilidade entre seeds e ausência de colapso direcional. Uma acurácia alta com cobertura quase nula ou concentrada em uma única direção não é evidência suficiente.

## Referências

[1]: https://arxiv.org/html/2512.12924v1 "Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals"

[2]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376 "Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment"

[3]: https://www.mdpi.com/2079-9292/15/6/1334 "Regime-Aware LightGBM for Stock Market Forecasting: A Validated Walk-Forward Framework with Statistical Rigor and Explainable AI Analysis"

[4]: https://scikit-learn.org/stable/modules/calibration.html "Probability calibration — scikit-learn User Guide"
