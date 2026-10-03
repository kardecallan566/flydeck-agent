# Evolution v3 — one frozen brain, 100–300 independent readouts, genuinely new candles

All changes remain in `feat/evolution-v2-survival-control-recent`.
The prior reports in `data/evolution/controlled-v2` and their 2000
recent observations (2026-09-25 to 2026-10-02 19:30 UTC) have **already
been examined**. They must not serve as a new independent test.

This update:
1. Calls `FlyVisualPredictionAgent.set_learning(False)` to disable the
   decision engine and episodic learner as well as the parent MaleCNS
   learning flag. The cache fingerprint includes a new inference version:
   the old 70k-candle neural cache MUST be rebuilt once.
2. Saves frozen, versioned `population_checkpoint.json` files with ALL
   100 agent weights and feature masks, input-history fingerprint, exact
   neural circuit fingerprint, and the original validation-only finalists.
   Checkpoints are saved BEFORE any recent-data adaptation.
3. Records an auditable `historical_audit_finalist_decisions.csv` (and
   `recent_sealed_finalist_decisions.csv` when requested) showing UP/DOWN/
   WAIT, probability, threshold, observation candle, settled outcome,
   scenario/snapshot odds and virtual capital.
4. Provides `flydeck-forward`, which replays two frozen populations on
   NEVER-BEFORE-INSPECTED 5m candles. A third inexpensive arm turns OFF
   the MaleCNS signal while keeping the SAME Fly-trained weights. This
   is a sensitivity check, not proof of causation.
5. Persists `forward_state.json` with paper equity, maximum equity,
   drawdown and halt status for continued tests across distinct periods.
   It never changes the frozen learning weights or places real bets.

## Step A — update this SAME branch and make new frozen checkpoints

From the FlyDeck repository directory in PowerShell:

```powershell
git fetch origin
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
py -3.11 -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_evolution.py tests/test_forward_eval.py

flydeck-evolve `
  --data data/cache/BNBUSDT_5m.csv `
  --circuit data/malecns/motion_visual.json `
  --rebuild-fly-cache `
  --compare-no-fly `
  --population 100 `
  --output data/evolution/controlled-v3
```

The corrected MaleCNS signal is calculated **once** over the old
70,000 candles, then reused across all agents. On later runs with
unchanged inputs, OMIT `--rebuild-fly-cache` and choose a distinct
output directory. The 100 readouts share one circuit; they do not
each instantiate 30,470 neurons.

You will get:
- `controlled-v3/with_fly/population_checkpoint.json`
- `controlled-v3/without_fly/population_checkpoint.json`
- `controlled-v3/ablation_report.json`
- per-arm `all_block_results.csv`, `historical_audit_finalist_decisions.csv`,
  `finalists.json`, `final_population.csv`, `summary.json`.

The historical blocks are useful for diagnostics, **not** independent proof:
the former validation/audit results have already been inspected.

## Step B — download candles AFTER the previously inspected 2,000

```powershell
flydeck-fetch-recent `
  --count 2000 `
  --after-history data/cache/BNBUSDT_recent_5m.csv `
  --output data/cache/BNBUSDT_next_5m.csv
```

The downloader **first filters out all previously inspected timestamps**, then
selects only newly closed, gap-free 5m candles. When fewer than
`--count` new candles exist it reports the **exact available count**
from the exchange response and the expected UTC time of the
requested target; it exits cleanly **without writing any file**.

For an early functional smoke test, explicitly opt in to a smaller
set instead of requesting nonexistent data:

```powershell
flydeck-fetch-recent `
  --count 100 `
  --available `
  --min-count 35 `
  --after-history data/cache/BNBUSDT_recent_5m.csv `
  --output data/cache/BNBUSDT_next_available_5m.csv
```

The actual CSV contains the new candles available up to 100, NEVER
previously inspected ones. Even the minimum 35 contains only 2
evaluable predictions after the 32-candle warmup and final unresolved
row. This is a plumbing smoke test, NOT a performance estimate.
If fewer than 35 exist, the downloader reports the count without
writing a CSV. Request 100 without `--available` when enough new
candles have closed; request 2000 for the real next-week holdout.

**Do not run the 100-candle smoke dataset as an independent 2000-candle
holdout later.** Once examined, those observations cease being unseen;
use a later non-overlapping dataset for the actual study.

The downloader verifies closed, gap-free, nonoverlapping 5m candles.
As the previous file ends October 2, 2026 19:30 UTC, this command
cannot legitimately produce a full 2,000 *new* candles until nearly
seven further days have passed. It refuses overlap rather than mixing
old observations into the sealed period. A smaller `--count 100` is
possible for a preliminary smoke check once at least 100 new candles
are available. Do not tune agents based on a smoke check and then
reuse that period as an unseen test.

The previous `BNBUSDT_recent_5m.meta.json` is tracked on this branch;
the large raw market CSVs were not present in the GitHub data directory
during the last review, so preserve your LOCAL raw CSVs. The downloader
uses Binance's public endpoint and requires no API key.

## Step C — evaluate all frozen agents on the genuinely NEW CSV

```powershell
flydeck-forward `
  --data data/cache/BNBUSDT_next_5m.csv `
  --after-meta data/cache/BNBUSDT_recent_5m.meta.json `
  --with-checkpoint data/evolution/controlled-v3/with_fly/population_checkpoint.json `
  --without-checkpoint data/evolution/controlled-v3/without_fly/population_checkpoint.json `
  --circuit data/malecns/motion_visual.json `
  --output data/evolution/forward-v3-next
```

`--after-file data/cache/BNBUSDT_recent_5m.csv` may replace
`--after-meta`. The command requires one of these to ensure the input
starts strictly after the earlier inspected window, and also rejects
overlap with the checkpoint's historical training data. It checks
that both checkpoints come from the same training setup and pins
the original circuit hash when available.

Three results are generated:
- `with_fly/`: frozen Fly-trained population with frozen v3 neural signal.
- `without_fly/`: frozen OHLCV-only population.
- `with_fly_signal_masked/`: the SAME frozen Fly-trained weights,
  with the neural input replaced with zero, without retraining.
  This counterfactual introduces an out-of-distribution input and is
  a diagnostic rather than a causal estimate.

`frozen_comparison.json` includes medians and descriptive differences.
Each arm writes `all_agents_forward.csv` for all agents,
`preselected_decisions.csv` for only the previously selected five,
`forward_summary.json` with input/model hashes,
and `forward_state.json` with cumulative virtual equity and
drawdown. Add `--trace-all` only if you want potentially large
per-decision logs for the entire population.

## Step D — continue a previous paper bankroll without replaying bets

On another strictly later distinct 5m CSV, run the same
`flydeck-forward` command with:
```powershell
--after-file data/cache/BNBUSDT_next_5m.csv `
--resume-root data/evolution/forward-v3-next `
--output data/evolution/forward-v3-next2
```

The `--resume-root` must contain all the same model arms. Each new
CSV skips the first 32 observations as causal warmup and the last
unresolved prediction; it persists bankroll and drawdown for all 100
agents instead of restarting equity at 100. No new learning occurs.

## What constitutes progress

Measure the ORIGINAL, validation-preselected finalists separately
from any candidates discovered *after* looking at recent data.
Treat family-level and all-agent median comparisons as exploratory.
A preselected model would need consistent new-period performance,
economically realistic multipliers and an appropriate multiple-testing
and time-dependence assessment before considering any real money.

The current default objective remains `binance-close-t+1`, not actual
PancakeSwap settlement. PancakeSwap's official lock/close prices use
Chainlink and its pools have variable payout ratios and a 3% treasury
fee. Pre-lock snapshots (if recorded) are scenarios, not guaranteed
final payouts. Without official epoch outcomes and actual payout
reconstruction, virtual returns cannot establish realizable profit.

Official docs:
- https://docs.pancakeswap.finance/play/prediction
- https://docs.pancakeswap.finance/play/prediction/prediction-faq

No real execution, wallet signing, automated agent promotion or
profitability guarantees are implemented.
