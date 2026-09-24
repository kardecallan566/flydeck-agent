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
