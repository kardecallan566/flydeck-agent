# FlyDeck Agent — BNB 5-Minute Prediction

## Objective

FlyDeck is no longer a trading agent. The primary task is a binary prediction task inspired by PancakeSwap Prediction:

> Given the current BNB/USD market state, decide whether the BNB price at the end of the next 5-minute round will be higher (`UP`) or lower (`DOWN`) than the round's reference price.

The agent does not buy or sell BNB. It does not manage a portfolio. It only observes, predicts, waits for the outcome, and learns from the result.

PancakeSwap Prediction uses rolling 5-minute rounds. A prediction is correct when the closing price is above the locked/reference price for `UP`, or below it for `DOWN`. citeturn0search0turn0search1

## Agent behavior

At each prediction opportunity the fly agent produces:

- `UP` — predicted price increase
- `DOWN` — predicted price decrease
- `WAIT` — confidence is insufficient; skip this round

`WAIT` is not a market action. It means the fly did not find enough evidence to make a prediction.

The agent is evaluated on:

- directional accuracy;
- confidence calibration;
- accuracy of entered predictions;
- number of skipped rounds;
- streaks of correct/incorrect predictions;
- performance across unseen time periods;
- computational cost.

The primary metric is **accuracy on entered predictions**, but it must always be reported together with coverage (`entered / available rounds`) so the agent cannot obtain a misleading score by predicting only extremely rarely.

## Why this task

This gives the connectome a concrete, repeatable behavioral objective without turning the fly into a conventional trading bot.

The market is the environment. BNB price movement is the sensory signal. The MaleCNS architecture is responsible for transforming observations into a behavioral decision.

## Biological direction

The project must use the MaleCNS as an agent architecture rather than as a generic recurrent reservoir.

The implementation should progressively preserve:

1. real MaleCNS neuron identities;
2. anatomical regions and neuron types;
3. directed synaptic connectivity;
4. neurotransmitter information where available;
5. excitatory/inhibitory effects;
6. recurrent state and temporal dynamics;
7. learning/plasticity mechanisms;
8. explicit sensory and behavioral interfaces.

Artificial input/output pools are temporary infrastructure only and must not be treated as biological mappings.

## Data protocol

The canonical training data is BNB/USD 5-minute historical price data. Every sample must be chronological. For each prediction round:

```text
reference price = price at prediction round start
             ↓
        fly observes
             ↓
      UP / DOWN / WAIT
             ↓
          5 minutes
             ↓
       outcome resolved
```

The training target is strictly derived from the future price relative to the reference price. No future information may enter the observation presented to the fly.

PancakeSwap notes that its interface uses real-time Binance/TradingView data while Chainlink is used for the prediction round's lock/end prices. Therefore the benchmark should distinguish between the market observation feed and the official prediction outcome feed when reproducing the PancakeSwap task. citeturn0search0

## Non-goals

The project does not currently aim to:

- execute trades;
- connect a wallet;
- place PancakeSwap bets automatically;
- optimize portfolio return;
- predict arbitrary crypto assets;
- simulate all ~166k MaleCNS neurons online;
- tune the fly into a black-box neural predictor through endless hyperparameter searches.

## Development order

1. Define the exact 5-minute BNB prediction environment.
2. Build a leak-free historical BNB dataset and evaluator.
3. Build a minimal behavioral fly interface with `UP/DOWN/WAIT`.
4. Replace artificial reservoir behavior with connectome-derived functional structure.
5. Incorporate neurotransmitter and anatomical information.
6. Add biologically motivated learning/plasticity.
7. Validate the fly on unseen BNB periods.
8. Only after the behavioral system works, consider broader environments.
