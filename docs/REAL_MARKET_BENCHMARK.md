# Real Market Benchmark

The real-market benchmark is intentionally separate from the synthetic market. Synthetic data remains the controlled laboratory for architecture experiments; real OHLCV data is used to test whether a learned policy generalizes outside the synthetic generator.

## Data source

The first implementation uses Binance Spot public OHLCV data through `GET /api/v3/klines`. No API key is required for public market data.

The downloader stores only the canonical fields needed by FlyDeck:

```text
timestamp,open,high,low,close,volume
```

Large datasets are kept outside Git under `data/real/` and are reproducible with the downloader.

## Download a long dataset

Example for BTC/USDT hourly candles:

```bash
python -m flydeck.finance_real_cli download \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2018-01-01T00:00:00Z \
  --output data/real/BTCUSDT_1h.csv
```

The downloader paginates the public endpoint and writes a single chronological CSV.

## Chronological evaluation

The benchmark uses:

```text
70% train
15% validation
15% unseen test
```

The splits are chronological, never shuffled. Each validation/test split retains 24 previous candles as lookback context so the existing feature encoder can calculate its rolling features without allowing future observations into the state.

Run:

```bash
python -m flydeck.finance_real_cli benchmark \
  --data data/real/BTCUSDT_1h.csv \
  --symbol BTCUSDT \
  --interval 1h
```

The benchmark trains V8 selected-action TD on the training history only, then evaluates validation and test data greedily.

## Decision diagnostics

For every test decision, the benchmark records:

- timestamp/index
- action
- close price
- current portfolio position ratio
- HOLD/BUY/SELL scores
- BUY-HOLD and SELL-HOLD score advantages
- forward close-to-close returns at 1, 3, 6, 12 and 24 candles

This lets us distinguish a policy that merely trades from a policy whose confidence is aligned with future market movement.

## What counts as progress

A better benchmark is not automatically a higher raw return. We want to see:

1. performance on unseen periods improve relative to HOLD and BUY&HOLD baselines;
2. drawdown remain controlled;
3. BUY decisions show better forward returns than HOLD decisions when BUY-HOLD advantage is high;
4. SELL decisions become meaningful instead of disappearing;
5. excessive consecutive BUY behavior decreases;
6. results remain stable across different chronological test periods.

Do not tune against the test split. If a change is made after inspecting test results, create a new experiment and validate it on a fresh unseen period.

## Reproducibility rule

Every benchmark report should record:

- dataset filename and source
- symbol and interval
- number of rows
- train/validation/test boundaries
- network seed
- environment costs and penalties
- architecture parameters
- action distribution
- return and drawdown
- HOLD and BUY&HOLD baselines
- decision-quality diagnostics

The raw real-market CSV should not be committed to Git. The code and benchmark configuration are the reproducible artifact.
