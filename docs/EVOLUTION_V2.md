# Evolution v2 — survival, MaleCNS ablation and fresh Binance holdout

This branch extends `feat/evolution-population-100-300`. It does not modify
the real-bet executor and **cannot place live bets**.

## 1. Download 2,000 **closed** 5m Binance candles

The Binance public market endpoint `data-api.binance.vision/api/v3/klines`
requires no key and returns at most 1,000 rows per HTTP request. The existing
`BinanceMarketDataProvider` automatically paginates the 2,002 requested rows.
Two spare rows cover a partially open candle and exchange-boundary timing.

```powershell
git fetch origin
git switch --track origin/feat/evolution-v2-survival-control-recent
py -3.11 -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_evolution.py tests/test_recent_candles.py

flydeck-fetch-recent `
  --count 2000 `
  --output data/cache/BNBUSDT_recent_5m.csv `
  --after-history data/cache/BNBUSDT_5m.csv
```

The downloader checks exactly 2,000 **fully closed** candles, unique,
chronological, no 5m gaps, recent freshness and strict non-overlap with the
existing 70k-candle CSV, and creates an adjacent `.meta.json` file.
It does not overwrite the old dataset. If it rejects overlap, use a smaller
`--count`, collect additional days, or choose an earlier historical cutoff.
Do **not** force overlapping data into an unseen test.

Two thousand 5m candles represent ~6.94 days; precisely seven days = 2016.
The last ~32 candles of early history are needed only as a causal context
warmup when evaluating a file that contains only new observations.

## 2. Run a same-seed controlled MaleCNS ablation

```powershell
flydeck-evolve `
  --data data/cache/BNBUSDT_5m.csv `
  --recent-data data/cache/BNBUSDT_recent_5m.csv `
  --recent-holdout 2000 `
  --circuit data/malecns/motion_visual.json `
  --compare-no-fly `
  --population 100 `
  --output data/evolution/controlled-v2
```

The output structure:

- `controlled-v2/with_fly/`: population of 100 lightweight learnable policies
  with a single shared cached 30,470-neuron MaleCNS feature vector.
- `controlled-v2/without_fly/`: 100 analogous policies without that signal,
  with the same seed, training blocks, fees and risk assumptions.
- `controlled-v2/ablation_report.csv` and `.json`: cohort distributions
  in validation, historical audit and optional recent holdout.
- `recent_sealed_results.csv` in **both** arms: all candidate results on
  the newer sealed set; no training or champion promotion on that set.
- `final_population.csv`: independent flags `risk_survived`, `profitable`,
  `survived` (= both), `statistical_evidence` (illustrative lower Wilson
  bound above fixed-scenario break-even), plus both-block metrics.

With 2000 recent candles only, no recent adaptation occurs; 32 early candles
are skipped as feature warmup. The last unresolved prediction is excluded.
This provides a genuinely new **chronologically later** dataset when the
`--after-history` download check passed. Reusing it to pick a revised
winner would invalidate its independence for the next revision.

To compare your old 100-agent run with the v2 survival definitions without
running the heavy shared MaleCNS cache again, reuse the existing feature
cache (input CSV and circuit hashes must be unchanged).

## 3. Interpret the survival flags correctly

- `risk_survived`: minimum entries, minimum coverage, and no drawdown halt.
- `profitable`: terminal virtual equity strictly above 100 for that block.
- `survived`: BOTH risk-survived and profitable; not live approval.
- `statistical_evidence`: only in a FIXED synthetic payout scenario, a
  simple 95% binomial Wilson lower bound exceeds the scenario's break-even
  probability. Correlated predictions and extensive model search reduce
  what this simple threshold can establish.

The comparison is a *descriptive controlled experiment*, not a statistical
proof of profitability. A single seed is a first check; selecting the
strongest of 100 agents introduces multiple-testing bias. Real PancakeSwap
performance still requires official Chainlink-resolved rounds, payout quotes
recorded **before** the decision, and transaction costs.

Use `--population 200` or `300` only after the first 100-agent ablation
works comfortably on your machine. This v2 version still shares one
MaleCNS network; it does **not** instantiate 100/200/300 full connectomes.
