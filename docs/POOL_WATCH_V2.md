# Pool Watch v2 — relatório visual e comparação entre consultas

O HTML foi reorganizado a partir do resultado real do primeiro teste:
US$ 10 aportados, US$ 0,60 de custos estimados e 30 dias. Uma pool
com APY de 17,11% pode ter rendimento bruto de aproximadamente
US$ 0,1406 no período, mas o valor bruto menos os custos seria
aproximadamente -US$ 0,4594. **Isso ainda não inclui risco de mercado,
perda impermanente, depeg, impostos nem preço real das transações.**

O novo painel apresenta PRIMEIRO o que interessa: orçamento, reserva,
prazo e APR mínimo para cobrir APENAS custos (73% nesse exemplo).
Depois separa V2 de V3, mostra o resultado bruto menos os custos em
tamanho maior, explica o estado do card, exibe TVL e taxa divulgada e
permite expandir os riscos. Resultados positivos na projeção NÃO
significam lucro realizável. O símbolo "stablecoin" não comprova o
contrato, auditoria nem manutenção da paridade.

## 1. Atualização e teste offline

Na branch existente:

```powershell
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
python -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_pool_watch.py tests/test_pool_monitor.py
```

Recrie um relatório em outro diretório (arquivos antigos são preservados):

```powershell
flydeck-pool-watch `
  --budget-usd 20 `
  --allocation-usd 10 `
  --roundtrip-cost-usd 0.60 `
  --days 30 `
  --min-tvl-usd 100000 `
  --limit 12 `
  --out-dir data/reports/pancake-dia-01

Start-Process data/reports/pancake-dia-01/report.html
```

Uma consulta NOVA usa dados públicos do DefiLlama. Não verificamos as
cotações em contratos on-chain ou rendimentos pessoais. O programa
recusa sobrescrever um `--out-dir` já existente.

## 2. Próxima atualização: comparação de APY e liquidez (já implementada)

Depois de algumas horas ou no dia seguinte, execute OUTRA consulta
com a opção `--previous` apontando para o JSON do relatório anterior:

```powershell
flydeck-pool-watch `
  --budget-usd 20 `
  --allocation-usd 10 `
  --roundtrip-cost-usd 0.60 `
  --days 30 `
  --min-tvl-usd 100000 `
  --limit 12 `
  --previous data/reports/pancake-dia-01/report.json `
  --apy-change-pp 2 `
  --tvl-drop-pct 20 `
  --out-dir data/reports/pancake-dia-02

Start-Process data/reports/pancake-dia-02/report.html
Start-Process data/reports/pancake-dia-02/changes.html
```

A segunda execução gera:

- `report.html`: visão principal com resumo de mudanças
- `report.json`: snapshot, metadados e **índice de todas as pools
  que passaram nos filtros** do agregador, mesmo se só 12 cards
  estiverem visíveis; isso evita confundir mudança no ranking com
  remoção de uma pool
- `pools.csv`: as pools exibidas para Excel e Sheets
- `changes.html`, `changes.json`, `changes.csv`: aumentos e
  quedas relevantes de APY, quedas de TVL e pools que passaram a
  entrar/sair dos filtros do agregador

Mudanças em APY usam **pontos percentuais absolutos**: de 10% para 7%
representa -3 p.p. O padrão dispara quando a mudança é de pelo menos
2 p.p. Uma queda de TVL >=20% também é destacada. Isso NÃO significa
fraude, ataque, quebra de paridade ou valorização futura: apenas
identifica valores alterados entre duas respostas de terceiros.

**Atenção aos relatórios legados:** se o `report.json` antigo não
contiver `monitoring_index`, só as pools exibidas naquele relatório
são comparáveis. A saída marca explicitamente esse escopo parcial.
Quando a primeira e a segunda consultas têm filtros de TVL diferentes,
entradas e saídas da lista não são interpretáveis como mudanças no
contrato. Todas as consultas são manuais, somente leitura e não
produzem notificações automáticas.

## 3. Modo de desenvolvimento sem acesso à internet

É possível testar usando o arquivo de demonstração:

```powershell
flydeck-pool-watch `
  --snapshot examples/defillama-pancake-SYNTHETIC.json `
  --out-dir data/reports/demo-poolwatch-v2
Start-Process data/reports/demo-poolwatch-v2/report.html
```

O HTML demonstra, em destaque, que os valores são **FICTÍCIOS**.
Nunca redistribua uma demonstração como se contivesse cotações reais.

## 4. Como usar comercialmente com pouco dinheiro

Crie relatórios INFORMATIVOS (não custódia/gestão de fundos): orçamento,
comparação entre pools, custos assumidos, riscos, dados de liquidez,
mudanças de rendimento de um dia para outro e histórico de alertas.
Os dados vêm de um agregador independente, podem estar defasados e
podem exigir revisão de termos de redistribuição para uso comercial.

Não ofereça retornos, aprovação de apostas ou afirmações de
segurança contratual; usuários devem verificar pool/token e custos
efetivos antes de mover fundos. Dados com APY anunciado mais alto
NÃO são ranqueados como os mais recomendados. A alternativa de
não investir faz parte do relatório.

Próximos módulos ainda NÃO implementados: verificação on-chain de
endereço de pool/token, modelo V3 específico à faixa, taxas reais do
usuário, alertas agendados e um painel de clientes autenticado.
O estudo v3/v4 dos 2000 candles segue independente do Pool Watch.
