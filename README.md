# FlyDeck Agent

A lightweight research agent whose concrete objective is to use the MaleCNS (*Drosophila* connectome) as the architecture of a small autonomous predictor.

## Current objective: BNB 5-minute prediction

At the beginning of a 5-minute round, observe the current BNB/USD market state and decide whether BNB will finish the round **UP** or **DOWN** relative to the round's reference price. The agent may also choose **WAIT** when it does not have enough evidence.

This follows the behavioral structure of PancakeSwap Prediction: rolling 5-minute rounds with UP/DOWN resolved from the reference/locked price versus the ending price. citeturn0search0turn0search1

There is no portfolio, trading strategy, wallet, or automatic bet execution in the core agent.

## The fly is the agent

The project is not intended to attach a connectome to an ordinary machine-learning model. MaleCNS should provide the internal architecture responsible for perception, temporal state, decision making and learning.

The implementation prioritizes real neuron identities, anatomy, neuron types, pathways, directed connectivity, neurotransmitter information, excitatory/inhibitory signaling, recurrent state, biologically motivated plasticity, a minimal BNB sensory interface, and a minimal UP/DOWN/WAIT behavioral interface.

The current 2048-neuron degree-core reservoir is a prototype extraction, not the final fly architecture.

## Learning loop

```text
BNB market
    |
    v
Sensory interface
    |
    v
MaleCNS agent
    |
    +---- UP
    +---- DOWN
    +---- WAIT
    |
    v
5-minute outcome
    |
    v
Learning / plasticity
    |
    +---- next round
```

The future outcome is never available when the prediction is produced.

## Evaluation

Every run reports total rounds, UP/DOWN/WAIT counts, coverage, accuracy on entered predictions, accuracy by confidence bucket, correct/incorrect streaks, chronological train/validation/test results, and computational cost.

Accuracy without coverage is not meaningful because WAIT is a valid behavior.

## Data

The canonical dataset is chronological BNB/USD 5-minute data. PancakeSwap documents that its interface uses real-time Binance/TradingView prices while Chainlink is used for the lock/end prices that determine the prediction result. Observation data and outcome data must therefore be treated as separate concepts when reproducing the task. citeturn0search0

## Out of scope

- BNB trading
- portfolio optimization
- wallet integration
- automatic PancakeSwap betting
- generic multi-asset prediction
- endless hyperparameter experiments
- pretending an arbitrary reservoir is biologically equivalent to the fly
- online simulation of the entire ~166k-neuron connectome

## Development order

1. Exact 5-minute BNB prediction environment.
2. Leak-free historical BNB dataset and evaluator.
3. Fly sensory and behavioral interfaces.
4. Functional MaleCNS architecture derived from anatomical/connectome data.
5. Neurotransmitter-aware synapses and excitatory/inhibitory dynamics.
6. Learning and plasticity inside the fly architecture.
7. Unseen-period BNB evaluation.
8. Resource profiling and optimization.

## Principles

- The behavior comes first.
- The fly is the agent, not a feature extractor.
- Biological structure must justify architectural decisions.
- No random architecture search as a substitute for understanding the connectome.
- No future information may enter a prediction.
- WAIT is a legitimate behavior.
- Accuracy must be reported together with coverage.
- Keep memory, CPU and dependency requirements small.
