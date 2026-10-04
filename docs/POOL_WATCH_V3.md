# Pool Watch v3 — pequenas variações, comparação temporal e cenários de risco

**Somente leitura.** Nenhum comando assina transações, conecta carteira,
recomenda depósitos nem garante renda. A fonte continua sendo um agregador
público independente. O FlyDeck preserva todos os checkpoints v3/v4 e os
2.000 candles prospectivos sem alteração.

## 1. Por que a consulta em um minuto não gerou alertas?

O exemplo gerado em 04/10/2026 mostra duas consultas com menos de
um minuto de diferença. Antes, a comparação imprimia apenas
"nenhuma variação atingiu os limites", ocultando pequenas mudanças
que de fato pudessem existir. A v3 agora informa:

- horário legível de ambas as consultas e **minutos entre elas**;
- aviso específico para consultas com menos de uma hora: o agregador
  pode ter devolvido um snapshot ainda não atualizado;
- pools coincidentes por identificador do agregador, não só pelo símbolo;
- mudanças pequenas abaixo do limite e tabela das até oito maiores
  diferenças de APY observadas, ainda que nenhum alerta tenha disparado;
- movimentos importantes, separados em quedas do APY divulgado,
  aumentos que precisam ser verificados e quedas no TVL;
- comparação ampla somente se ambos os arquivos tiverem o índice
  completo e os mesmos filtros de TVL. Relatórios legados são parciais.

APY é medido em **pontos percentuais** entre consultas, e TVL em
variação percentual. **Não** extrapole uma diferença de um minuto
para desempenho diário, valorização ou rentabilidade. Mesmo uma pool
que some da consulta pode apenas não ter passado pelos filtros do
agregador, não ter sido incluída pela fonte ou estar em uma antiga
lista limitada a 12 cards.

## 2. Atualize e teste

No PowerShell com a venv ativada:

\`\`\`powershell
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
python -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_pool_watch.py tests/test_pool_monitor.py tests/test_pool_watch_v3.py
\`\`\`

O comando continua o mesmo, mas o relatório passa a conter novas
informações. **Sempre use um diretório novo.** Para comparar com
os relatórios já gerados, substitua o caminho em \`--previous\` pelo
seu arquivo real:

\`\`\`powershell
flydeck-pool-watch \`
  --budget-usd 20 \`
  --allocation-usd 10 \`
  --roundtrip-cost-usd 0.60 \`
  --days 30 \`
  --min-tvl-usd 100000 \`
  --limit 12 \`
  --stress-pct 30 \`
  --previous data/reports/pancake-dia-02/report.json \`
  --apy-change-pp 2 \`
  --tvl-drop-pct 20 \`
  --out-dir data/reports/pancake-v3-dia-03

Start-Process data/reports/pancake-v3-dia-03/report.html
Start-Process data/reports/pancake-v3-dia-03/changes.html
\`\`\`

Caso ainda não tenha um \`report.json\` anterior, remova a linha
\`--previous ...\` e use um diretório de saída único para a
primeira consulta. Se ambas as consultas usam o mesmo feed e
têm as mesmas pools, a tabela mostra variações pequenas mesmo
com zero alertas. Depois repita a consulta em um intervalo maior
para observar eventuais mudanças.

## 3. Novo: cenários de preço para V2

Cada card **V2** agora contém uma tabela expansível com hipóteses
de variação do preço de **somente um dos tokens**, enquanto o outro
mantém seu valor em dólar. O padrão testa -30%, -10%, 0%, +10% e
+30% (ou ajuste o choque principal com \`--stress-pct\`, de 1 a 90).

Para cada hipótese, o programa compara:

- valor de simplesmente guardar os dois tokens inicialmente 50/50;
- valor da posição LP de produto constante V2 sem rendimentos;
- valor LP acrescentando o rendimento bruto HIPOTÉTICO do APY
  divulgado pelo agregador e subtraindo o custo total ASSUMIDO;
- diferença entre LP e manter os tokens; perda impermanente
  percentual relativa a manter os tokens.

A rentabilidade do LP não é equivalente ao aumento do preço do
token: o pool reequilibra a exposição, e taxas podem não compensar
perda impermanente ou queda de preço. O modelo **não simula V3**,
cotações reais, preços dos dois tokens variando simultaneamente,
taxas dinâmicas, slippage variável, impostos ou risco do contrato.
O rendimento bruto em dólar é mantido fixo em todos os cenários
apenas para isolar o risco de preço, **não como previsão**.

Agora há também \`stress_v2.csv\`, além de
\`report.html\`, \`report.json\`, \`pools.csv\` e, quando
uma consulta anterior é fornecida, \`changes.html\`,
\`changes.json\` e \`changes.csv\`.

Para US$ 10 aportados e US$ 0,60 de custos, uma pool cujo APY
anunciado seja apenas 10% geraria por projeção linear cerca de
US$ 0,082 brutos em 30 dias. Mesmo SEM qualquer variação de
preço, isso representaria uma diferença negativa de cerca de
US$ 0,518 frente a manter os tokens. Se um dos tokens cair,
o patrimônio LP pode ter uma perda em dólares ainda maior.

## 4. Uso futuro e próximos passos

O sistema já permite produzir um relatório interpretável para
investigação pessoal ou uma prestação de serviço informativa
com dados públicos. Para qualquer divulgação comercial,
respeite termos de uso/licenciamento de dados do agregador,
mostre fontes e datas, e não anuncie essas análises como
consultoria regulada, certeza de lucro ou auditoria de contratos.

Próximos avanços ainda NÃO implementados:
- verificação da identidade dos contratos de pool e tokens por
  fonte oficial e RPC somente leitura;
- estimativas de gas atualizadas com transparência de rede;
- modelo de V3 específico à faixa de preços do usuário;
- snapshots históricos locais para relatórios semanais;
- alertas agendados opcionais sem gestão de carteira.

O FlyDeck de Prediction segue em paper trading; estes módulos
não modificam seu teste independente de 2.000 candles.
