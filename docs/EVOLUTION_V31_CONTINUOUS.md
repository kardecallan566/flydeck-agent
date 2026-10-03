# FlyDeck v3.1 — diagnóstico e avaliação contínua

Branch única: `feat/evolution-v2-survival-control-recent`. Os dois
`population_checkpoint.json` de `controlled-v3` permanecem congelados.
Nenhuma mudança habilita apostas reais.

## 1. Analise AGORA seus resultados existentes, sem precisar de novos candles

No PowerShell, na raiz do repositório:

```powershell
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
python -m pip install -e ".[evolution]"
python -m pytest

flydeck-diagnose `
  --run data/evolution/forward-v3-smoke `
  --output data/evolution/forward-v3-smoke-diagnostic.json
```

A auditoria de dados já examinados calcula:
- Quantos agentes não entraram. A acurácia mediana *entre agentes ativos*
  exclui os que ficaram sempre em WAIT; a antiga métrica os contava
  como 0% de acerto.
- Resultados separados dos 5 finalistas escolhidos na validação.
- Comparação pareada dos mesmos 100 pesos com e sem entrada neural;
  nos cinco finalistas, mede mudança de UP/DOWN/WAIT no mesmo horário.
- Brier score por probabilidade `p_up` para os finalistas, incluindo
  WAIT, com referência neutra de 0,25 (probabilidade fixa de 50%).
  Menor Brier significa probabilidades mais próximas dos resultados.
  As observações são correlacionadas entre agentes; isto NÃO é teste
  de significância nem prova de causalidade.
- Se o número de oportunidades permitia atingir o mínimo de entradas.

O relatório do primeiro teste de 95 candles é útil como diagnóstico,
mas JÁ FOI EXAMINADO; não é um novo holdout.

## 2. Baixe novos candles em lotes quando existirem pelo menos 35

```powershell
flydeck-fetch-recent `
  --count 100 `
  --available `
  --min-count 35 `
  --after-history data/cache/BNBUSDT_next_available_5m.csv `
  --output data/cache/BNBUSDT_batch02_5m.csv
```

O downloader continua recusando dados futuros, lacunas e sobreposição.
Depois, concatene o lote fresco ao arquivo dos 95 candles:

```powershell
flydeck-append-candles `
  --base data/cache/BNBUSDT_next_available_5m.csv `
  --new data/cache/BNBUSDT_batch02_5m.csv `
  --output data/cache/BNBUSDT_cumulative_02.csv
```

O novo comando conserva cada candle, exige adjacência de 5 minutos e
recusa sobrescrever fontes. Produz também
`BNBUSDT_cumulative_02.manifest.json` com hashes SHA-256,
quantidade de linhas, limites de horário e proveniência.
Na próxima coleta, use o último *lote fresco* em `--after-history`
e o CSV cumulativo mais recente em `--base`. Nunca inclua os mesmos
candles duas vezes.

## 3. Execute novamente os modelos CONGELADOS sobre toda a série acumulada

```powershell
flydeck-forward `
  --data data/cache/BNBUSDT_cumulative_02.csv `
  --cumulative-manifest data/cache/BNBUSDT_cumulative_02.manifest.json `
  --after-meta data/cache/BNBUSDT_recent_5m.meta.json `
  --with-checkpoint data/evolution/controlled-v3/with_fly/population_checkpoint.json `
  --without-checkpoint data/evolution/controlled-v3/without_fly/population_checkpoint.json `
  --circuit data/malecns/motion_visual.json `
  --output data/evolution/forward-v3-cumulative-02
```

**NÃO use `--resume-root` com `--cumulative-manifest`.** O modo
cumulativo reavalia todos os candles da primeira observação até a
última, com capital inicial 100 uma única vez. Evita perder 32 candles
de contexto em cada novo lote e mantém o circuito neural causal
através da mesma sequência. Não altera os pesos do checkpoint.
O cache MaleCNS é reaproveitado apenas para CSV/circuito idênticos.

A versão anterior também descartava uma previsão já resolvida por
um limite exclusivo. A correção faz com que 95 candles fechados
tenham exatamente **62 oportunidades avaliáveis** após 32 candles
de contexto e a última observação sem rótulo.

As saídas agora incluem acurácia mediana apenas entre agentes com
entradas, proporção de agentes que nunca entraram e três comparações
simples sobre as MESMAS observações: sempre UP, sempre DOWN e
sempre WAIT. Os dois primeiros têm retorno somente num cenário
*hipotético* de pagamento fixo, taxas e gas configurados.

## 4. Cientificamente, não confunda fluxos

O CSV cumulativo inclui candles que você já examinou e serve para
acompanhar paper trading e depurar o modelo. NÃO chame esse fluxo
de teste independente. Para demonstrar qualquer melhoria aprendida
a partir dessas observações, preserve outra janela de 2.000 candles
totalmente posteriores e não examinados. Não altere os thresholds
ou selecione candidatos usando o teste e depois reutilize-o como
confirmação.

O alvo atual é `binance-close-t+1`. O PancakeSwap usa preços
lock/close do oráculo e multiplicadores de pools variáveis, com
taxa de 3%; preços de mercado Binance e pagamento fixo 2x são
apenas aproximações. Fonte:
https://docs.pancakeswap.finance/play/prediction/prediction-faq

Próximos checkpoints dependentes de dados externos: importar rodadas
oficiais e pagamentos finais, registrar snapshots anteriores às
decisões, analisar custo de gas e dependência temporal com bootstrap
por blocos, calibrar probabilidades SOMENTE no desenvolvimento e
pré-registrar políticas para janelas futuras. Não há execução real
nem evidência suficiente de rentabilidade.
