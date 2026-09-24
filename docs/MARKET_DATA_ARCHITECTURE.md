# FlyDeck market-data architecture

FlyDeck now separates market-data acquisition from the neural system.

## Flow

```text
Binance REST ─────┐
                  │
CSV / replay ─────┼──> MarketDataProvider
                  │           │
                  │           ▼
                  │    canonical OHLCV
                  │           │
                  │           ▼
                  │      validation
                  │           │
                  │           ▼
                  │      local cache
                  │           │
                  └───────────┤
                              ▼
                       MarketDataService
                              │
                              ▼
                    BNBPredictionDataset
                              │
                              ▼
                  Retina / MaleCNS / CX / MB
```

## What was extracted from the OpenStock architecture

The useful idea is the separation of concerns, not the Next.js application itself:

- provider-specific API access lives in `data/binance_provider.py`;
- the agent consumes a canonical `MarketCandle`/`MarketDataset` contract;
- `MarketDataService` orchestrates fetching, cache loading and refresh;
- CSV persistence is isolated in `data/market_cache.py`;
- data quality is checked before it reaches the prediction benchmark;
- the same contract can later be used for Binance live data, historical CSV replay, or another provider.

OpenStock documents a provider-oriented market-data integration around Finnhub and warns that provider limitations and rate limits affect the data available to the application. FlyDeck applies the same architectural principle while keeping its Python neural core independent from OpenStock's AGPL-3.0 codebase.

## Current data-quality protections

The canonical loader validates:

1. timestamp ordering;
2. duplicate timestamps;
3. positive OHLC prices;
4. OHLC consistency (`high`/`low` envelopes);
5. non-negative volume;
6. expected candle spacing for the selected interval;
7. empty datasets.

The benchmark refuses BNB 5m CSVs containing interval gaps. This prevents a silent missing-candle problem from becoming a neural signal.

## Compatibility

The loader accepts both:

- `timestamp,open,high,low,close,volume`;
- legacy `open_time,open,high,low,close,volume`.

This fixes the earlier schema mismatch where one benchmark expected `open_time` while the stored datasets used `timestamp`.

## Why this matters for the neural weaknesses

The recent FlyDeck failures (UP/LONG collapse, unstable Q-values, churn and weak out-of-sample behavior) cannot be fixed by adding more neural complexity while the input pipeline remains opaque or inconsistent.

This layer gives the next phases a stable foundation:

- deterministic replay;
- explicit temporal boundaries;
- reproducible cached datasets;
- source provenance;
- gap detection;
- easy provider substitution;
- clean separation between data acquisition and the biological model.

The neural system should therefore be evaluated against the same canonical dataset regardless of whether the candles came from Binance, a CSV replay, or a future provider.
