# FlyDeck experimental v4 — investigate first, train locally with matched controls

**Branch:** `feat/evolution-v2-survival-control-recent`. All original v3
`population_checkpoint.json` files remain unchanged. Do NOT delete,
replace, retrain, or promote them using the already-seen 300 candles.
All figures below use Binance next-close direction and ILLUSTRATIVE fixed
2× payout (3% fee / zero gas), NOT actual PancakeSwap profits.

## Research findings that motivate this opt-in experiment

The existing 300-candle rolling test contains **265 resolved directional
outcomes** after skipping the 32-candle causal context and two neutral
labels; there were 267 timestamp opportunities. UP occurred **57.36%**.
The Fly-trained population had median active-agent accuracy **43.55%**
and median paper return **−2.106%**; the OHLCV-only control had
**43.48%** and **−1.424%**. Each population had 3/100 profitable agents;
neither has statistical evidence above the scenario break-even threshold.
The five old Fly finalists made 53 UP and 413 DOWN entries, and
only **3 out of 1325** paired finalist actions changed when their Fly
input was masked. The masked neural input is out of distribution; it is
a sensitivity test, not a causal treatment-effect estimate.

These are **already-inspected observations**, including previously
examined data. Never use them again as an independent validation set.

## Step 1 — diagnose history vs already-inspected forward observations

Update from the same branch in PowerShell:

```powershell
git switch feat/evolution-v2-survival-control-recent
git pull --ff-only
python -m pip install -e ".[evolution]" pytest
python -m pytest tests/test_v4_research.py tests/test_evolution.py tests/test_forward_eval.py

flydeck-investigate `
  --history data/cache/BNBUSDT_5m.csv `
  --with-checkpoint data/evolution/controlled-v3/with_fly/population_checkpoint.json `
  --without-checkpoint data/evolution/controlled-v3/without_fly/population_checkpoint.json `
  --forward-run data/evolution/forward-v31-repaired-300 `
  --fly-cache data/cache/fly_shared_features.npz `
  --output data/evolution/v4-v3-diagnostic.json
```

This strictly offline command checks:
- SHA-256 of the original historical CSV against **both** v3 checkpoint
  fingerprints, and the selected forward checkpoint hashes.
- Actual UP/DOWN and neutral-label rates per chronological train,
  development, validation and historical-audit block, with no
  retrospective fit of the forward target.
- UP prior estimated from **train + development only**. Compares this
  frozen reference and a simple 50%-UP reference with the *already-seen*
  forward up rate and its hypothetical Brier score.
- Effective model intercept `bias` across all 100 agents and among the
  five previously validation-selected finalists.
- Historical MaleCNS signal mean, standard deviation, 5/50/95th
  percentiles, zeros, descriptive signal-to-next-outcome correlation,
  and the actual effective neural feature weight (`weights[-1] ×
  mask[-1]`) in each Fly finalist. The supplied cache's original
  market SHA-256, circuit SHA-256, version and timestamps MUST match.
- Paired same-weight on/masked signal decision disagreement and Brier
  scores on the original finalists' time-aligned predictions,
  including WAIT outcomes.

If the historical CSV or cache file differs from the v3 checkpoint,
this tool fails closed. Never pretend that an unrelated history or
stale cache belongs to the frozen model.

## Step 2 — experimental v4 learning: ONLY past resolved labels

The v3 training/evaluation remain unchanged by default.

v4 adds three explicit, independently configurable controls to
`flydeck-evolve`:

```text
--class-balance-alpha 0.75
--class-weight-cap 1.5
--bias-l2-multiplier 4.0
```

In chronological training/development blocks, an observation's
binary label enters a two-class counter ONLY when its settlement time
has been reached. The next learning event's weight is the inverse
of the *PREVIOUSLY SETTLED* class frequency, clipped to
`[1/1.5, 1.5]`, blended 75% with unit weight. Counts begin with
one pseudocount per class in each block. Intercept regularization is
four times the existing l2 coefficient; the remaining features and
checkpoint feature schema are unchanged.

This explicitly tests the **historical DOWN-bias hypothesis**.
It does **not** guarantee better probabilities or performance. Class
balancing may actually damage probability calibration, since the
training target is reweighted. DO NOT treat `p_up` as a calibrated
economic win probability without a later calibration fitted ONLY on
unseen-to-training development data.

The same seed, chronological blocks, fee/risk assumptions, and original
70k-candle history are used for matched Fly/no-Fly v4 experiments.
The ~30,470-neuron MaleCNS is computed ONCE on the original history,
then cached; the cheap 100 readouts are independently evolved.

## Step 3 — quick local test with 100 agents and one seed

```powershell
.\scripts\run_v4_experiment.ps1 `
  -Seeds @(42) `
  -Population 100 `
  -OutputRoot data/evolution/v4-single-seed
```

This script refuses the wrong branch, missing original v3 checkpoints,
missing market/circuit files, duplicate seed IDs and EXISTING output
directories. It runs the targeted pytest tests (omit only with
`-SkipTests`) and records exact settings in each
`seed-42/experiment_plan.json`. Training logs are preserved in
`seed-42/training.log`, and each seed gets two frozen candidate
checkpoints and a descriptive `ablation_report.json`.

If your original neural cache was lost/stale, explicitly use
`-RebuildFlyCache` (expensive); it rebuilds only once on the first
seed, never independently for each lightweight policy.
Do not use the 300-candle forward cache in place of the original
70k-candle historical cache.

## Step 4 — longer multi-seed experiment on your Windows PC

AFTER the one-seed smoke check works:

```powershell
.\scripts\run_v4_experiment.ps1 `
  -Seeds @(143, 244, 345) `
  -Population 100 `
  -Threads 4 `
  -OutputRoot data/evolution/v4-three-more-seeds
```

This runs **six matched** independent-readout trials sequentially
(three seeds × Fly / OHLCV-only), reusing the original verified neural
cache. Increase to `-Population 300` only if time and RAM allow,
using a SEPARATE output root. An RTX3050 GPU is not required for the
readout evolution, and the system does not assume one neuron
circuit per readout.

**This is a bounded experiment, NOT a promise to train for exactly
seven days.** It ends when the listed seeds finish; it never downloads
future candles, changes the original checkpoints or places real
trades. Preserve all failures and adverse results, not just
promising training seeds. Keep the computer ventilated and use the
power/performance limits appropriate to its hardware.

## Step 5 — true prospective assessment remains separate

Do NOT use `BNBUSDT_300_repaired.csv` to select the best hyperparameters
or report a v4 "out-of-sample victory"; this stream has already been
examined and motivated the v4 hypothesis. Validation and audit on
the old historical 70k are also no longer fresh independent evidence.

Freeze the settings and candidate IDs using ONLY historical
development/validation. A fair future comparison runs frozen v3 and
frozen v4 checkpoints on an entirely **new**, later, timestamp-verified
2,000-candle holdout, with a matched OHLCV control, the identical
scoring assumptions, the original five finalists tracked separately,
and no decisions changed after seeing the holdout.

The present objective `binance-close-t+1` is a *proxy* for the
PancakeSwap game. Official settlement requires historical oracle
lock/close prices and causally observed pre-lock pool distributions;
actual multipliers vary and gas/fees matter. Neither the historical
nor current simulated returns justify real-money trading.

Official game docs:
https://docs.pancakeswap.finance/play/prediction/prediction-faq
