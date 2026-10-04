# FlyDeck Agent

A lightweight artificial agent whose neural substrate is derived from the complete male *Drosophila* nervous-system connectome (MaleCNS).

## Current objective: BNB 5-minute prediction

The fly has one job:

> At the start of each five-minute round, observe the BNB market state and decide whether the next five-minute reference-to-close movement will be `UP`, `DOWN`, or `WAIT` when the fly does not have enough evidence.

This behavior is inspired by the five-minute Prediction game on PancakeSwap. The FlyDeck agent does **not** trade BNB, manage a wallet, or execute blockchain transactions.

```text
BNB market history
        ↓
   sensory system
        ↓
   MaleCNS agent
        ↓
  UP / DOWN / WAIT
        ↓
      5 minutes
        ↓
    real outcome
        ↓
 neural learning
```

## What makes it a fly agent

The MaleCNS is the computational substrate, not merely a random reservoir used by a conventional trading model. The project progressively preserves:

- real neuron identities;
- anatomical organization and neuron types;
- directed connectivity;
- neurotransmitter information;
- excitatory/inhibitory structure;
- recurrent state and temporal dynamics;
- biologically motivated plasticity;
- explicit sensory and behavioral interfaces.

The current compact circuit is an implementation bridge while the functional MaleCNS mapping is developed. Artificial input/output pools are not considered biological claims.

## Behavioral rules

- `UP`: the fly predicts a higher BNB close after five minutes.
- `DOWN`: the fly predicts a lower BNB close after five minutes.
- `WAIT`: confidence is insufficient, so the fly skips the opportunity.

The evaluation reports both **accuracy on entered predictions** and **coverage**. This prevents a fly from appearing successful simply by predicting extremely rarely.

## Data protocol

The benchmark uses chronological BNBUSDT 5-minute candles. At candle `t`, the fly receives only information from candles `<= t`. The target is determined from the next candle:

```text
reference = close[t]
future    = close[t + 1]

future > reference → UP
future < reference → DOWN
future = reference → no directional outcome
```

No future candle is included in the sensory input.

### Exact PancakeSwap target mode

The vNext benchmark can replace the Binance next-close proxy with the actual
PancakeSwap BNB Prediction result: Chainlink Locked Price -> Closed Price. The
alignment uses only a fully closed Binance candle available before the chosen
decision cutoff, so the partially formed candle around lock cannot leak into the
features.

Download settled round structs through a read-only BNB Chain JSON-RPC endpoint:

```bash
flydeck-pancake-rounds \\
  --rpc-url YOUR_BNB_RPC_URL \\
  --last 75000 \\
  --output data/cache/pancake_bnb_rounds.csv
```

Run baselines and the FlyDeck visual agent on the same frozen protocol:

```bash
flydeck-benchmark-vnext \\
  --data data/cache/BNBUSDT_5m.csv \\
  --pancake-rounds data/cache/pancake_bnb_rounds.csv \\
  --decision-lead-seconds 30 \\
  --circuit data/malecns/motion_visual.json \\
  --fly-ablations \\
  --context 32 --purge 1 \\
  --min-validation-entries 500 --min-coverage 0.10
```

Without --pancake-rounds, the benchmark keeps the original Binance close[t] ->
close[t+1] proxy so old experiments remain reproducible. The round downloader is
read-only and does not sign transactions, connect a wallet, or place bets.

The repository contains a dependency-free loader and a small downloader for public Binance market data. Historical datasets should be stored locally and never committed when they are large.

## Run

Build/load a MaleCNS circuit first, for example:

```bash
python -m flydeck.malecns_cli build \
  --annotations data/malecns/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --weights data/malecns/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --output data/malecns/degree_core_2048.json
```

Download a small BNB sample:

```bash
python -m flydeck.bnb_prediction_cli \
  --circuit data/malecns/degree_core_2048.json \
  --download data/real/BNBUSDT_5m.csv \
  --limit 1000
```

Run a local historical benchmark:

```bash
python -m flydeck.bnb_prediction_cli \
  --circuit data/malecns/degree_core_2048.json \
  --data data/real/BNBUSDT_5m.csv
```

Train the visual agent with causal survival learning:

```bash
python -m flydeck.bnb_visual_cli \
  --data data/real/BNBUSDT_5m.csv \
  --circuit data/malecns/motion_visual.json \
  --survival --lives 3 --rounds 1000 \
  --confidence 0.15 --exploration 0.30 \
  --min-exploration 0.05 --wait-streak 8
```

During survival training, the agent loses a life only after an entered UP/DOWN
prediction is resolved by the following candle and is incorrect. To avoid the
degenerate policy of waiting forever, epsilon exploration decays toward the
configured minimum and a directional probe is forced after a configurable WAIT
streak. The report separates raw WAIT decisions from exploratory actions and
must be judged using accuracy, coverage, and death rate together; the legacy
survival-rate field is not a profitability metric.

Tests:

```bash
python -m pytest
```

## Non-goals

The FlyDeck Agent is not currently intended to:

- execute trades;
- connect a wallet;
- place PancakeSwap bets automatically;
- manage portfolio capital;
- optimize trading profit;
- predict arbitrary assets;
- simulate all ~166k MaleCNS neurons online;
- become a black-box predictor through endless hyperparameter searches.

## Development direction

1. Establish a leak-free BNB five-minute behavioral environment.
2. Make the sensory interface causal and efficient.
3. Use MaleCNS structure as the agent architecture.
4. Map anatomical and functional pathways instead of inventing arbitrary output semantics.
5. Incorporate neurotransmitter effects.
6. Add biologically motivated plasticity and learning.
7. Evaluate on unseen BNB periods with accuracy, coverage and resource usage.
8. Only then consider other environments.

## Principles

- The behavior comes from the agent, not a portfolio wrapper.
- Biological structure should have a documented computational role.
- No future information may enter the fly's sensory system.
- `WAIT` is a valid behavior.
- Prefer structural explanations over blind parameter searches.
- Keep memory, compute and dependencies small.

## PancakeSwap economics and dedicated sealed holdout collection

[Income and research workflow](docs/EARN_AND_SEALED.md):
offline V2 liquidity-vs-HOLD comparison, risk/cost sensitivity,
Prediction break-even without bets, and a separate SHA-256-chained
incremental collector that NEVER scores the prospective 2000-candle
dataset. No wallet, real trades, guaranteed APR or live-profit claims.

## Experimental v4 — historical drift audit and opt-in balanced learning

The [v4 experimental guide](docs/EXPERIMENTAL_V4.md) shows how to:
audit the original ~70k-candle historical distribution against the
already-inspected 300-candle forward run; verify the MaleCNS cache and
actual final-candidate effective neural weights; train optional v4
resolved-label-only class weighting and intercept regularization; run
matched Fly/no-Fly multi-seed experiments locally on Windows without
overwriting frozen v3 models. A subsequent truly NEW window, not the
300 previously inspected candles, must evaluate any v4 hypothesis.

## Next test — repair the 105-candle gap before the weekly monitor

See [NEXT_TEST_300_CANDLES.md](docs/NEXT_TEST_300_CANDLES.md)
for the complete, lossless process: recover the exact historical
03:25–12:05 UTC gap, assemble 300 contiguous candles without overwriting
existing files, and evaluate all original frozen agents. This is an
already-inspected rolling diagnostic, **not** a fresh sealed holdout.

## Evolution v3.1 — continuous research and honest audits

Start with [EVOLUTION_V31_CONTINUOUS.md](docs/EVOLUTION_V31_CONTINUOUS.md):
offline analysis of the existing 95-candle smoke run (no new candles required);
WAIT-safe accuracy and probability calibration; append-only, strictly
non-overlapping candle streams; and cumulative replay of the original
frozen model checkpoints without double-counting paper capital.

## Evolution v3 — frozen prospective test

See [EVOLUTION_V3_FORWARD.md](docs/EVOLUTION_V3_FORWARD.md) to reproduce the 100-agent controlled ablation with a fully frozen shared MaleCNS, export all model checkpoints and evaluate NEW Binance candles without retraining or real-money execution.
