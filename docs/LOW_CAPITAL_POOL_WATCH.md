# FlyDeck Pool Watch — painel público de pools para orçamento inferior a US$ 20

**Disponível agora:** consulta pública de APYs agregados do DefiLlama,
filtro de PancakeSwap na BNB Chain, cálculo ilustrativo de custo versus
rendimento bruto, relatórios estáticos em HTML, CSV e JSON. Não conecta
carteira, assina transações, aposta, compra tokens ou recomenda depósitos.

**Aviso:** O DefiLlama é um AGREGADOR INDEPENDENTE. As taxas podem estar
desatualizadas ou incompletas; nenhum APY é uma cotação oficial ou um
rendimento confirmado para a sua carteira. Pools V3 dependem da faixa
de preço e deixam de receber algumas recompensas ao sair dela.
Nem APY elevado nem stablecoins significam investimento seguro.

## 1. Instale na mesma branch

No PowerShell, com a venv ativa, na raiz do repositório:

```powershell
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
python -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_pool_watch.py tests/test_earn_planner.py
```

## 2. Teste OFFLINE com números inventados, sem internet

Um arquivo de exemplo está incluído para verificar HTML e exportação:

```powershell
flydeck-pool-watch `
  --snapshot examples/defillama-pancake-SYNTHETIC.json `
  --budget-usd 20 `
  --allocation-usd 10 `
  --roundtrip-cost-usd 0.60 `
  --days 30 `
  --out-dir data/reports/pancake-demo-01

Start-Process data/reports/pancake-demo-01/report.html
```

**Números da demonstração NÃO são pools nem APYs vigentes.**

## 3. Busque os dados públicos atuais

```powershell
flydeck-pool-watch `
  --budget-usd 20 `
  --allocation-usd 10 `
  --roundtrip-cost-usd 0.60 `
  --days 30 `
  --min-tvl-usd 100000 `
  --limit 12 `
  --out-dir data/reports/pancake-primeira-pesquisa

Start-Process data/reports/pancake-primeira-pesquisa/report.html
```

Escolha um NOVO `--out-dir` a cada consulta; o programa recusa
sobrescrever relatórios anteriores. O `--roundtrip-cost-usd` é seu
custo hipotético TOTAL de entrar E sair, incluindo gas, swaps e
slippage que você mesmo estimou. Não é obtido da rede.

`report.html`: página estática responsiva, que você pode enviar
a um possível cliente como relatório de PESQUISA (deixe visíveis
data, fonte e alertas). `pools.csv`: comparação no Excel ou Sheets.
`report.json`: insumo para um painel futuro. Verifique os termos de
uso da fonte antes de redistribuir comercialmente dados de terceiros.

**Exemplo de economia:** investimento de US$ 10 com APY agregado de
10% por 30 dias renderia apenas cerca de US$ 0,082 BRUTOS com uma
projeção linear simplificada. Com custo total de US$ 0,60, a diferença
já seria NEGATIVA (cerca de -US$ 0,518) antes de qualquer risco.
Para cobrir SÓ US$ 0,60 de custo em 30 dias, seriam necessários
aproximadamente **73% ao ano** nesse modelo — ainda sem perdas de
preço, depeg ou impermanent loss. Para capital pequeno, NÃO investir
pode ser o resultado correto.

## 4. Como utilizar para tentar gerar receita sem arriscar o capital

O foco comercial inicial pode ser um **relatório informativo
personalizado**, não gestão de carteira:
- Comparar pools específicas solicitadas pelo cliente, com TVL,
  data, fonte e yield agregado, ressalvando erros/staleness.
- Estimar o efeito de aportes pequenos, custos assumidos e horizonte.
- Apresentar cenários adversos usando o comando
  `flydeck-earn-plan lp` somente para pools **V2 50/50**.
- Entregar CSV/HTML com riscos e a alternativa de não abrir posição.

Nenhum relatório deve afirmar que o FlyDeck constatou oportunidade
lucrativa, que APY da V3 seria realizável, ou que algum investimento
está seguro. Não ofereça gestão de fundos, recomendações reguladas,
custódia ou garantias de rendimento.

## 5. O que NÃO faz ainda

Não há execução de ordens; não há verificação da autenticidade do
contrato/token, cobrança de impostos, cálculo personalizado de
liquidez concentrada V3, coleta de taxas reais da carteira,
rentabilidade REAL, remuneração ao usuário, monitoramento contínuo
de 24 horas ou alertas automáticos. A ferramenta executa uma
consulta ÚNICA por comando. Não deixe executando continuamente
para simular renda automática. A avaliação independente dos 2.000
candles fica em `flydeck-seal-merge`, separada deste scanner.

Fontes e checagem manual:
- https://yields.llama.fi/pools
- https://defillama.com/yields
- https://pancakeswap.finance/liquidity/pools
- https://docs.pancakeswap.finance/earn/earn-faq/farming-faq
- https://docs.pancakeswap.finance/trade/trading-faq/swap-faq
