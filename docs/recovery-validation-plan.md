# Plano de recuperação e validação do agente MaleCNS

## Decisão principal

A matriz de ablação não justifica continuar adicionando módulos ao agente atual. No conjunto de teste, o baseline original obteve 46,377% de accuracy, 1,314% de coverage, retorno líquido de -2,2410% e profit factor de 0,1503. A versão `current_temporal_crypto_event` obteve 5,854% de accuracy, 1,952% de coverage, retorno líquido de -4,5964% e profit factor de 0,1105.

A conclusão operacional é clara: **a versão atual deve ser congelada como hipótese reprovada**, e o baseline deve ser congelado como referência. Nenhuma nova camada deve ser adicionada antes de corrigirmos o protocolo de medição e criarmos barreiras contra regressões.

Não existe garantia honesta de que um agente de trading alcançará 60–70% de acurácia ou lucro futuro. O que podemos garantir é um processo que impeça o projeto de gastar horas em candidatos claramente piores, reduza o risco de overfitting e só promova mudanças que apresentem evidência independente e economicamente relevante.

## Por que os testes anteriores não produziram progresso confiável

O projeto reutilizou a mesma série histórica para testar várias ideias. Isso cria risco de *data snooping*: depois de muitas tentativas, algum resultado pode parecer bom por acaso. White descreve esse problema como a reutilização dos mesmos dados para inferência e seleção de modelos, e propõe um teste de realidade que compara a melhor especificação encontrada com um benchmark, em vez de tratar a melhor tentativa como evidência isolada [1].

O Deflated Sharpe Ratio foi desenvolvido para corrigir a inflação causada por múltiplos testes e por retornos não normais. Portanto, o número de hipóteses testadas precisa ser registrado e levado em conta; não é válido escolher a melhor configuração e ignorar as tentativas que falharam [2].

A validação walk-forward atual é melhor do que uma divisão aleatória, mas ainda não é suficiente para uma busca iterativa longa. Estudos recentes apontam que métodos purged e combinatorial purged cross-validation podem reduzir o risco de vazamento entre observações próximas e fornecer uma avaliação mais estável em séries com dependência temporal e mudanças de regime [3].

## Garantias de processo que serão adotadas

### 1. O teste final será intocável

O período final será separado uma única vez. Ele não poderá ser usado para escolher parâmetros, ajustar thresholds, decidir entre variantes ou interpretar uma falha e tentar novamente. O arquivo de resultado desse período deverá ser gravado com hash e a configuração deverá ser registrada antes da execução.

Toda decisão de desenvolvimento usará apenas treino, validação e um conjunto de desenvolvimento com múltiplas janelas temporais. O teste final só será aberto depois que uma candidata passar por todos os gates anteriores.

### 2. Cada tentativa terá um identificador

Cada execução registrará:

- commit do código;
- hash do dataset;
- hash do circuito MaleCNS;
- seed;
- parâmetros completos;
- custo, slippage e regra de execução;
- número acumulado de tentativas;
- resultado por janela e por regime.

Assim, a correção estatística poderá considerar quantas alternativas foram realmente experimentadas.

### 3. Nenhum benchmark longo começará sem um pré-teste

Antes de percorrer 70 mil candles e 30 mil neurônios, a candidata deverá passar por um conjunto pequeno e determinístico. O pré-teste deve verificar:

- ausência de lookahead;
- alinhamento entre sinal, execução e retorno futuro;
- custos aplicados somente quando a posição muda;
- coerência entre label, horizonte e métrica;
- posições limitadas ao intervalo permitido;
- comportamento em dados constantes, tendência sintética, reversão sintética e choque sintético;
- ausência de NaN, overflow e crescimento de memória;
- tempo por candle e tempo total estimado.

Uma candidata que falhar em qualquer item será descartada sem benchmark completo.

### 4. O baseline terá um contrato de não regressão

Uma mudança não será promovida apenas por aumentar coverage ou accuracy de treino. Para substituir o baseline, deverá cumprir simultaneamente:

- retorno líquido no teste de desenvolvimento não pior que o baseline por mais de 0,25 ponto percentual;
- profit factor não inferior ao baseline em pelo menos 3 de 5 janelas;
- drawdown não superior ao baseline em mais de 0,50 ponto percentual;
- accuracy e hit rate não podem cair simultaneamente;
- nenhum fold com equity final abaixo de 0,90 sem justificativa previamente definida;
- desempenho positivo ou claramente menos negativo depois dos custos em pelo menos 4 de 5 janelas.

Esses limites são gates de engenharia, não uma promessa de que o sistema será lucrativo.

## Protocolo em fases

### Fase A — Auditoria barata da medição

**Objetivo:** provar que a avaliação mede exatamente o que pretendemos.

Primeiro, manteremos somente o baseline e uma política de posição fixa. A mesma decisão será avaliada com horizonte de 1 candle, custos idênticos e uma função única para `accuracy`, `hit_rate`, `coverage` e retorno econômico.

A avaliação deverá separar três conceitos:

- **accuracy direcional:** direção correta entre as entradas;
- **hit rate econômico:** trades com retorno líquido positivo depois de custo;
- **retorno da carteira:** composição dos retornos com mudança de posição e custo de turnover.

A comparação só será considerada válida quando o baseline reproduzir o resultado já conhecido dentro de uma tolerância numérica pequena. Se a reprodução falhar, nenhum modelo será treinado ou comparado.

**Custo esperado:** segundos a poucos minutos. **Gate:** todos os testes unitários e de invariantes passam.

### Fase B — Diagnóstico sem aprendizado

**Objetivo:** descobrir se há sinal direcional antes de usar memória, plasticidade ou política contínua.

Executaremos o MaleCNS congelado em janelas curtas e calcularemos:

- distribuição de `p_up`, `p_down` e `p_wait`;
- matriz de confusão por regime;
- accuracy condicional ao confidence threshold;
- calibration curve, Brier score e ECE;
- retorno bruto e líquido por ação;
- autocorrelação do sinal e estabilidade entre janelas.

O agente só poderá avançar se existir uma faixa de threshold com evidência repetida de edge líquido. Se nenhuma faixa superar o acaso depois do custo, o problema será de representação ou alinhamento do sinal, não de política de risco.

**Custo esperado:** minutos. **Gate:** pelo menos uma faixa de threshold estável em múltiplas janelas, sem usar o teste final.

### Fase C — Reabilitação do sinal direcional

**Objetivo:** melhorar a decisão UP/DOWN sem permitir que a política econômica crie sinais.

A política Crypto Event será removida da geração de direção. O MaleCNS continuará produzindo o lado da operação. A camada econômica poderá apenas:

- reduzir exposição;
- bloquear uma entrada sem edge líquido;
- escolher tamanho de posição dentro de limites;
- aplicar stop, cooldown e limite de drawdown.

A memória temporal será usada primeiro como confirmação binária ou multiplicador de exposição. Ela não poderá inverter a direção do MaleCNS. Essa restrição permite testar se a memória ajuda ou apenas injeta ruído.

O teste será feito com uma pequena matriz pré-registrada de no máximo seis hipóteses. Não será permitido alterar thresholds após observar resultados da validação sem contar uma nova tentativa.

**Custo esperado:** minutos por hipótese no conjunto reduzido. **Gate:** superar o baseline em pelo menos 3 de 5 janelas de desenvolvimento.

### Fase D — Horizontes múltiplos somente se houver evidência

**Objetivo:** testar 1, 3 e 6 candles sem misturar decisões incompatíveis.

Cada horizonte terá seu próprio relatório e não será escolhido depois da decisão por uma heurística de memória. Primeiro mediremos o edge de cada horizonte separadamente. Só depois avaliaremos uma combinação, usando uma regra pré-registrada de seleção.

O horizonte escolhido deverá ser aquele com maior retorno líquido ajustado ao custo dentro do treino e da validação, nunca aquele que apresentar a maior accuracy isolada. Se nenhum horizonte tiver edge consistente, a política retornará WAIT.

**Custo esperado:** dezenas de segundos a poucos minutos por candidato após as fases anteriores. **Gate:** o horizonte escolhido supera o baseline sem aumentar drawdown e turnover de forma desproporcional.

### Fase E — Validação robusta e teste final

**Objetivo:** estimar se o ganho sobrevive a diferentes períodos.

Usaremos cinco ou mais janelas temporais de desenvolvimento com purging e embargo equivalentes ao maior horizonte de previsão. A candidata será avaliada em cada janela sem atualização durante o período de validação.

A promoção dependerá de uma análise de estabilidade, não de um único número:

- mediana e pior caso do retorno líquido;
- mediana e pior caso do profit factor;
- máximo drawdown;
- CVaR;
- turnover e custo total;
- accuracy e hit rate por regime;
- intervalo bootstrap para retorno médio;
- Probabilistic Sharpe Ratio e Deflated Sharpe Ratio quando houver série suficiente;
- comparação com baseline por bootstrap pareado ou teste de realidade quando houver múltiplas candidatas.

Somente depois disso o teste final será aberto uma única vez.

**Custo esperado:** benchmark completo, mas executado uma única vez para uma ou duas candidatas aprovadas.

## Política de parada

O processo será interrompido imediatamente quando ocorrer qualquer uma destas condições:

1. a candidata perde mais de 0,25 ponto percentual de retorno líquido contra o baseline em duas janelas de desenvolvimento;
2. o profit factor fica abaixo de 1,0 em todas as janelas avaliadas;
3. o turnover cresce sem aumento proporcional do retorno bruto;
4. a accuracy de treino sobe, mas a validação cai além do limite de tolerância;
5. o agente concentra mais de 90% das entradas em uma direção ou em um único regime;
6. a política aumenta coverage e piora simultaneamente retorno, drawdown e profit factor;
7. o pré-teste detectar desalinhamento de horizonte ou vazamento temporal.

Quando uma candidata falhar, não serão feitos ajustes improvisados para “salvá-la” no mesmo conjunto. Ela será marcada como reprovada e qualquer nova hipótese receberá outro identificador.

## Critério realista de sucesso

A meta de 60–70% de accuracy não será usada como único objetivo. Em mercados líquidos e ruidosos, accuracy alta pode ser economicamente inútil se vier com custo, baixa cobertura ou drawdown elevado.

O primeiro marco de sucesso será recuperar, em dados de desenvolvimento, uma estratégia que seja pelo menos tão boa quanto o baseline em retorno e risco. O segundo será obter retorno líquido não negativo depois de custos em várias janelas. Somente então a accuracy será otimizada dentro da faixa de operações economicamente viáveis.

Se nenhuma variante superar o baseline após um número controlado de hipóteses, a conclusão correta será que o sinal atual não demonstrou edge suficiente nessa base de 5 minutos. Nesse caso, devemos mudar a informação de entrada ou o ativo/horizonte, e não continuar aumentando a complexidade do MaleCNS.

## Implementação imediata recomendada

A próxima alteração não deve ser uma nova arquitetura. Deve ser um **harness de validação fail-fast** com:

1. teste de reprodução do baseline;
2. manifesto imutável da tentativa;
3. pré-teste de 500–2.000 candles;
4. cinco janelas de desenvolvimento;
5. gates automáticos de não regressão;
6. saída incremental por janela e por variante;
7. execução completa somente após aprovação dos gates;
8. arquivo separado para o teste final intocável.

Essa mudança reduz o desperdício de tempo e torna cada resultado auditável. Ela não garante lucro, mas garante que o projeto pare de interpretar regressões como progresso.

## Referências

[1]: https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf "A Reality Check for Data Snooping"

[2]: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"

[3]: https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 "Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment"

[4]: https://arxiv.org/html/2512.12924v1 "A Rigorous Walk-Forward Validation Framework for Market Prediction"
