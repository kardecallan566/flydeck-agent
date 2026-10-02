# FlyDeck Evolution — 100/200/300 agents (research-only)

Branch: `feat/evolution-population-100-300`. Based on `feat/audit-vnext-p0`,
which contains market-data loaders, official PancakeSwap round RPC/alignment,
baseline comparisons, and risk/validation code. No real-bet executor is
enabled or invoked. The old GUI executor remains historical on another branch.

## Overview

- **100/200/300** independent, lightweight online logistic policies split into 10 families,
  with different causal features, thresholds, weights, and learning rates.
- One optional **shared, frozen** MaleCNS visual signal computed ONCE per CSV and
  saved as a fingerprinted cache. This is population exploration around MaleCNS
  features, **not** 300 independent full neural circuits. Without `--circuit`,
  screening uses OHLCV-only features.
- Every agent has independent online readout weights and parameters. Online
  learning occurs only when the relevant next-candle or oracle outcome has
  become available. Each block resets *virtual* equity to 100 to compare
  survival fairly; the neural readouts keep learning across development blocks.
- 3 historical train blocks, 2 evolution blocks, 1 validation block and
  1 diagnostic historical audit block (default 10k candles each). A small
  1-index gap is omitted at the first block boundary and decisions whose
  outcome would cross a boundary are excluded.
- After every **development** block, keep the best 20% with family diversity,
  mutate descendants to refill the population. Freeze the entire population
  at validation. Select at most 5 finalists ONLY from historical validation.
  Audit and recent holdout **never select a replacement**.
- Export EVERY candidate's per-block metrics, the final population, lineage,
  finalists with parameters, and optional recent-sealed results.
- Uses NumPy vectorization over agents. O(N candles × N agents × 13 features).
  No repeated 70k-candle MaleCNS pass per candidate.

## Install

From repo root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e . numpy pytest
python -m pytest tests/test_evolution.py
```

## Fast OHLCV-only population (start here)

```powershell
flydeck-evolve --data data/cache/BNBUSDT_5m.csv --population 100 --output data/evolution/p100
```

Switch to `--population 200` or `300` only after a successful first run.
Increase `--min-entries` if the selected agents barely participate.
Seven default 10k historical blocks require roughly 70k candles.
This is **research**: the latest historical audit has already been examined,
so it is not a fresh proof of positive performance.

## Optional MaleCNS feature

```powershell
flydeck-evolve --data data/cache/BNBUSDT_5m.csv --circuit data/malecns/motion_visual.json --population 100 --output data/evolution/fly100
```

First run builds `data/cache/fly_shared_features.npz`; subsequent runs reuse
it when the CSV/circuit hashes and context match. If the CSV is updated,
use `--rebuild-fly-cache`.

## Download additional fresh Binance candles

```powershell
python -m flydeck.market_data_cli --help
```

Or, in Python:
```python
from flydeck.bnb_prediction_data_runner import download_bnb_5m_csv
download_bnb_5m_csv("data/cache/BNBUSDT_new_5m.csv", limit=10000)
```

**Important:** 10,000 five-minute candles cover ~34.7 days, not a week.
One week is 2,016 candles. Data should be chronological, gap-free,
and have no overlap with the sealed holdout after merging/deduping.
The CLI rejects a sealed recent holdout that overlaps the old history.

```powershell
flydeck-evolve --data data/cache/BNBUSDT_5m.csv --recent-data data/cache/BNBUSDT_new_5m.csv --recent-holdout 2016 --population 100 --output data/evolution/recent100
```

Recent data before the final 2,016 observations can adapt agents with delayed
feedback; the final 2,016 are sealed and do not update weights or select winners.
For strict temporal independence, prefer a recent CSV whose full timeframe is
later than the historical training CSV. Do not repeatedly inspect a sealed
holdout while tuning parameters; collect a new prospective period afterward.

## Exact PancakeSwap outcomes (optional, preferred for application validity)

Use the pre-existing read-only RPC tool:

```powershell
flydeck-pancake-rounds --rpc-url $env:BNB_RPC_URL --last 12000 --output data/cache/pancake_rounds.csv
flydeck-evolve --data data/cache/BNBUSDT_5m.csv --pancake-rounds data/cache/pancake_rounds.csv --population 100 --output data/evolution/pancake100
```

Only matching settled oracle rounds are eligible, with causal decision cutoffs.
**Economic returns need pre-lock payout snapshots.** If you have recorded
them prospectively, CSV header is
`epoch,timestamp_ms,gross_up,gross_down`.
Pass `--odds-snapshots path.csv`. A snapshot must predate the decision
by no more than 120 seconds. Never substitute final pool totals for pre-lock
odds; that would leak future information. Without snapshots, equity and survival
use the explicit, **hypothetical** 2x gross odds / 3% fee scenario, not historical
realized profit. Do not use such results for live promotion.

## Files

- `all_block_results.csv`: all generations and all agents by block.
- `final_population.csv`: all surviving population slots, validation/audit metrics.
- `lineage.csv`: ancestry and hyperparameters of every created candidate.
- `finalists.json`: up to five candidates selected on validation, with weights.
- `recent_sealed_results.csv`: results for every agent on independent recent data.
- `summary.json`: full settings, target, evidence limitations, and finalists.

## Interpretation and constraints

- Population search across 100–300 variants introduces multiple-testing bias.
  A highest-performing agent is **not** established as profitable.
- An agent may be best in the survival *scenario* without any true trading edge.
- Official Chainlink outcomes and pre-decision payout snapshots are necessary
  for claims about performance on PancakeSwap. Gas, latency, wallet execution,
  fees and real slippage must be independently measured before considering live.
- Selection optimizes illustrative equity less drawdown while imposing minimum
  entries and coverage; it is not an estimate of success probability.
- The shared MaleCNS circuit is frozen during population screening to fit PCs
  with 16GB RAM. An optional later step can fine-tune only finalists with
  truly independent full neural states, outside this minimal MVP.
- **No browser automation, wallet signing or real-money orders are included**.
