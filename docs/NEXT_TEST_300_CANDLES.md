# Next checkpoint: recover the 105-candle gap and run the 300-candle diagnostic

**Stay on** `feat/evolution-v2-survival-control-recent`. Keep the original
100-agent v3 checkpoints FROZEN. This is an *already-inspected rolling*
paper-trading diagnostic, not another independent sealed validation set.

## Why the append error is correct

`BNBUSDT_cumulative_02.csv` already exists and must not be overwritten.
Also, the first 95-candle file ended at **2026-10-03 03:25 UTC** (last
close) while the next 100-candle file starts at **2026-10-03 12:10 UTC**
(first open). Exactly **105 five-minute candles** are missing between
03:25 and 12:05 UTC. Simply renaming the output would NOT solve the gap.

First inspect the existing file and its manifest. If it already contains
a correct gap-repaired 300-candle dataset, do NOT download/reappend:

```powershell
$old = Import-Csv data/cache/BNBUSDT_next_available_5m.csv
$new = Import-Csv data/cache/BNBUSDT_batch02_5m.csv
$old.Count
$new.Count
$old[-1].timestamp
$new[0].timestamp
if (Test-Path data/cache/BNBUSDT_cumulative_02.manifest.json) {
  Get-Content data/cache/BNBUSDT_cumulative_02.manifest.json -Raw
}
if (Test-Path data/cache/BNBUSDT_cumulative_02.csv) {
  $prior = Import-Csv data/cache/BNBUSDT_cumulative_02.csv
  [pscustomobject]@{
    Count = $prior.Count
    FirstOpenMs = $prior[0].timestamp
    LastOpenMs = $prior[-1].timestamp
  }
}
```

## Exact bounded repair: never overwrite source files

```powershell
git pull --ff-only
python -m pip install -e ".[evolution]"
python -m pytest tests/test_forward_gap.py tests/test_forward_stream.py

flydeck-fill-gap `
  --base data/cache/BNBUSDT_next_available_5m.csv `
  --later data/cache/BNBUSDT_batch02_5m.csv `
  --output data/cache/BNBUSDT_gap_02_5m.csv
```

The new downloader uses Binance's bounded `startTime`/`endTime`
historical API. It requests ONLY the exactly missing interval and
verifies the entire 300,000-ms timestamp grid, fully closed candles,
duplicates and source-file hashes. For the shown terminal outputs,
the result must contain exactly **105 gap candles**. It produces a
`.gap.json` provenance file and refuses existing output files.

```powershell
flydeck-append-candles `
  --base data/cache/BNBUSDT_next_available_5m.csv `
  --new data/cache/BNBUSDT_gap_02_5m.csv `
  --output data/cache/BNBUSDT_95_plus_gap.csv

flydeck-append-candles `
  --base data/cache/BNBUSDT_95_plus_gap.csv `
  --new data/cache/BNBUSDT_batch02_5m.csv `
  --output data/cache/BNBUSDT_300_repaired.csv
```

Expected total: **95 + 105 + 100 = 300** contiguous candles. Original
source files and the existing `BNBUSDT_cumulative_02.csv` are unchanged.
For 300 candles, the fixed forward replay can score **267** fully
resolved next-candle labels (32 causal warmup, 1 unresolved last candle).

If any output listed above already exists, inspect it and choose a
fresh name; never delete or overwrite research source files merely
to make a rerun pass.

## Run the 300-candle diagnostic against original frozen checkpoints

```powershell
flydeck-forward `
  --data data/cache/BNBUSDT_300_repaired.csv `
  --cumulative-manifest data/cache/BNBUSDT_300_repaired.manifest.json `
  --after-meta data/cache/BNBUSDT_recent_5m.meta.json `
  --with-checkpoint data/evolution/controlled-v3/with_fly/population_checkpoint.json `
  --without-checkpoint data/evolution/controlled-v3/without_fly/population_checkpoint.json `
  --circuit data/malecns/motion_visual.json `
  --fly-cache data/cache/fly_forward_repaired_300_v3.npz `
  --output data/evolution/forward-v31-repaired-300

flydeck-diagnose `
  --run data/evolution/forward-v31-repaired-300 `
  --output data/evolution/forward-v31-repaired-300-diagnostic.json
```

**DO NOT use `--resume-root` with `--cumulative-manifest`.** Full replay
begins with equity=100 exactly once and maintains the full unbroken neural
state over the combined series. The unique `--fly-cache` path avoids
a stale 95-candle feature cache fingerprint.

Read the diagnostic as an EXPERIMENT, not a new independent holdout:
- Count agents with `entered = 0` instead of calling them 0%-accurate.
- Compare active-only accuracy and Brier scores, including WAIT
  predictions, for the validation-preselected five.
- Compare exactly paired neural-on vs neural-masked decisions at the
  same timestamp, and always-UP/always-DOWN baselines.
- Preserve all three arms and their provenance; do not promote the
  agent with the highest return from the now-examined 300 candles.

## Next week's data collection

After the second 100-candle file, the next *unseen* candle opens at
2026-10-03 20:30 UTC. Every new collection must start EXACTLY at the
previous lot's next candle. The latest-window command cannot promise
contiguity after a long pause; v3.2 now REJECTS missing historical
intervals rather than silently returning a recent 100-candle tail.
Use `flydeck-fill-gap` to recover bounded missed data when necessary.

This is one-week PROSPECTIVE MONITORING, not one-week automatic model
training. Keep checkpoint weights unchanged, accumulate and timestamp
new candles, and reserve an entirely later, never-reviewed window for
a genuinely independent evaluation. None of the synthetic 2x payout
results constitutes PancakeSwap profitability.
