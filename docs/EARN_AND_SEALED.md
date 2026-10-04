# FlyDeck: capital-preserving PancakeSwap economics + untouched 2000-candle collector

Research only. None of the commands connects a wallet, trades, estimates
actual live pool APR or earns income automatically.

## What the uploaded research reports actually show

- 4 matched v4 seeds `42,143,244,345` completed.
- Development-only historical UP frequency: about 50.16% versus
  57.36% in the ALREADY inspected 300-candle sample.
- Of the 100 frozen v3 Fly candidates, 97 had negative intercepts.
- Shared MaleCNS signal std ~0.0652 and same-window descriptive signal
  vs next-UP correlation approximately -0.0168. Five finalist neural
  effective weights are small; do not equate ~30k neurons with evidence
  of real profitable prediction.
- v4 seed42 on the inspected 300-candle period: 4/100 Fly agents
  profitable, median return -2.138% and active accuracy 43.517%;
  OHLCV-only 3/100 profitable, median -2.491% and 44.153%.
- Its five Fly finalists switched only 3 of 1325 paired
  UP/DOWN/WAIT actions when the Fly feature was masked; their mean
  Brier scores were 0.253946 on and 0.253939 masked, worse than
  the 0.25 fixed 50/50 reference. **There is no measured betting edge.**
- Across the four v4 seeds, Fly minus OHLCV historical AUDIT
  median virtual equity is positive in each seed but Fly minus OHLCV
  VALIDATION median virtual equity is negative in 3 of 4 seeds.
  DO NOT select on the already-inspected audit period.

## Immediate PancakeSwap earning options

**There is no guaranteed daily cash generation by Prediction.** Its real
oracle target and pooled payout differ from the FlyDeck Binance
next-candle proxy. The official Prediction FAQ states a 3% treasury fee,
variable reward ratio set by pool sizes, whole-stake loss on an incorrect
or tie result and on-chain claim gas. With present results, **no real
Prediction betting**. Old PancakeSwap Simple Staking and third-party
Position Manager guides may be archived; check current official UI,
contract, permitted network and withdrawal conditions before any
capital commitment.

There are two capital-based alternatives to INVESTIGATE, not promises:
- **V2/V3 liquidity provision** may earn trading fees; eligible Farms may
  additionally earn CAKE, subject to actual live emissions. Pools
  require two assets. A V3 position earns in-range and can go out of
  range. Both have market, smart-contract and impermanent-loss risks.
- **Active currently offered single-asset staking** may offer rewards,
  but check live status: historical Simple Staking pages are ARCHIVED.
  A posted headline APR is not a guaranteed redeemable net cash rate.

For zero/minimal trading capital, monetize a SERVICE instead:
offer clearly labeled analytics, read-only public market monitoring,
risk summaries or a comparison calculator to other LP researchers.
Do not sell claims of certain BNB price direction. Check relevant
consumer, tax, advertising and financial-service rules before marketing.

Official docs:
https://docs.pancakeswap.finance/earn/yield-farming/how-to-use-farms
https://docs.pancakeswap.finance/earn/earn-faq/farming-faq
https://docs.pancakeswap.finance/play/prediction/prediction-faq

## New offline economics planner

Install new entry points after pulling the SAME research branch:

```powershell
git pull --ff-only
python -m pip install -e ".[evolution]"
python -m pytest tests/test_earn_planner.py tests/test_sealed_collect.py
```

For a purely hypothetical **V2 50/50 constant-product LP** with one
token unchanged, one token declining 20%, **ASSUMED, UNVERIFIED** 10%
annual trading-fee APR, 0% farming, $3 total gas and $1 slippage:

```powershell
flydeck-earn-plan lp `
  --capital-usd 100 `
  --token-a-change-pct -20 `
  --token-b-change-pct 0 `
  --swap-fee-apr-pct 10 `
  --farm-apr-pct 0 `
  --days 30 `
  --total-gas-usd 3 `
  --slippage-usd 1 `
  --output data/evolution/earn-v2-scenario-01.json
```

It compares net LP versus just holding each original 50/50 token
and versus **not depositing capital**. Since LP liquidity is
exposed to token price movements, any trading fees can be overwhelmed
by the underlying token drawdown and impermanent loss. Unlike V2,
V3 concentrated positions require a separate out-of-range model.
The APR here is USER INPUT, not a PancakeSwap live quote.

Inspect WHY 52% win rate under hypothetical fixed 2x/3%-fee
still gives only a marginal, *unsupported* edge:

```powershell
flydeck-earn-plan prediction `
  --estimated-win-probability 0.52 `
  --gross-payout 2 `
  --treasury-fraction 0.03 `
  --gas-fraction-of-stake 0.005
```

This tool gives a break-even rate and expected profit IF every
assumption were valid. Our present research DOES NOT establish a 52%
true win probability, a guaranteed prelock 2x payout, or real
economic advantage. An EV computation is NOT a bet recommendation.

## Correct separate collector for the already saved 79 candles

Your first prospective tranche is:
`data/cache/BNBUSDT_v4_sealed_batch01.csv`, 79 rows, with SHA-256:

`8751B346F570E19F7B6B4C08EB8D229297074C52E614FE2B8E0281A39943AA1C`

**Do NOT run prediction, inspect future outcomes, train or select
on these 79 candles.** Save file hashes and provenance.

When new candles exist, use the LAST raw tranche as the
`--after-history` boundary:

```powershell
flydeck-fetch-recent `
  --count 100 `
  --available --min-count 35 `
  --after-history data/cache/BNBUSDT_v4_sealed_batch01.csv `
  --output data/cache/BNBUSDT_v4_sealed_batch02.csv
```

Then use the SEPARATE collection-only tool. Unlike
`flydeck-append-candles`, this manifest does not falsely
label the dataset "already inspected":

```powershell
flydeck-seal-merge `
  --base data/cache/BNBUSDT_v4_sealed_batch01.csv `
  --new data/cache/BNBUSDT_v4_sealed_batch02.csv `
  --output data/cache/BNBUSDT_v4_sealed_cumulative02.csv `
  --target-count 2000
```

Produces a SHA-256-chained
`BNBUSDT_v4_sealed_cumulative02.sealed.json` manifest.
Neither component is overwritten. This tool does not look at
outcomes, train, or make forecasts; "unseen" still depends on
YOU not using these candles elsewhere to influence the research.
Subsequent merges use the latest `...cumulativeNN.csv`
as --base and the newest fresh batch as --new. If the gap
exceeds the requested most recent 100, the downloader refuses
silently skipped candles; use the bounded `flydeck-fill-gap`
with the known next batch instead, and do not run predictions
on any part of the sealed source chain.

**At EXACTLY 2,000** rows, freeze the file and
`*.sealed.json` hash chain. Only then run a ONE-TIME
`flydeck-forward` on the identical data for original v3
and all four already-frozen v4 populations, each with its own
output directory and no tuning from the prospective results.
Use `--after-file data/cache/BNBUSDT_batch02_5m.csv`
as the earliest last-inspected boundary, NOT
`--cumulative-manifest` (that option is for the separately
INSPECTED rolling diagnostic). After evaluation, archive the
full reports, including unfavorable results and oracle mismatch.

## Next improvements, *not* claimed as implemented

Read-only official BNB round and quote collector; pre-decision pool
snapshots; actual post-claim and gas accounting; V3
concentrated-liquidity modeling; LP APR and contract-status
verification against authenticated live sources; a non-custodial
customer dashboard and explicit tax/safety disclaimers; causal
MaleCNS normalizing from development only; a truly new registered
model after observing the prospective 2000 candles.
