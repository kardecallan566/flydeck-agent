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

The agent is evaluated on directional accuracy, confidence calibration, entered-prediction accuracy, coverage, streaks, unseen-period performance, and computational cost.

## Biological direction

The MaleCNS must be treated as the agent architecture rather than as a generic recurrent reservoir. The implementation should preserve real neuron identities, anatomical regions, neuron types, directed connectivity, neurotransmitter information, excitatory/inhibitory effects, recurrent state, learning/plasticity, and explicit sensory/behavioral interfaces.

Artificial input/output pools are temporary infrastructure only and are not biological mappings.

## Data protocol

The canonical training data is chronological BNB/USD 5-minute price data. For each round:

```text
reference price = price at round start
             ↓
        fly observes
             ↓
      UP / DOWN / WAIT
             ↓
          5 minutes
             ↓
       outcome resolved
```

The future close must never be present in the observation used to produce the prediction.

PancakeSwap documents that its Prediction rounds run every 5 minutes and that UP/DOWN is resolved from the locked/reference price versus the end price. It also documents separate real-time and oracle price feeds, so the benchmark must distinguish the observation feed from the official outcome definition. citeturn0search0

## Non-goals

The project does not currently aim to execute trades, connect wallets, place PancakeSwap bets automatically, optimize portfolio return, predict arbitrary assets, simulate every MaleCNS neuron online, or tune a black-box predictor through endless parameter searches.

## Development order

1. Build the exact 5-minute BNB prediction environment.
2. Build a leak-free historical BNB dataset and evaluator.
3. Give the fly a real behavioral interface: `UP/DOWN/WAIT`.
4. Derive functional structure from MaleCNS instead of using an arbitrary degree-core reservoir.
5. Incorporate anatomy, neuron types, neurotransmitters, and excitatory/inhibitory signaling.
6. Add biologically motivated learning/plasticity.
7. Validate on unseen BNB periods.
8. Only after the behavioral system works, consider broader environments.
