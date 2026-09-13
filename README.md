# FlyDeck Agent

A lightweight agent built with efficiency as a first-class goal.

The long-term goal is to explore whether structural principles from the *Drosophila* connectome can help build capable agents with low computational and memory requirements.

## Current direction

The project starts **without the fly connectome**. The current goal is to stress the small recurrent agent on a difficult sequential problem before introducing connectome-derived structure.

The first serious domain is **synthetic crypto-market decision making**. This is intentionally not a live trading bot: the agent operates on generated market episodes with virtual capital and learns from portfolio changes.

## Financial experiment

The synthetic market produces OHLCV-like candles and switches between different regimes:

- upward and downward trends
- sideways markets
- high volatility
- reversals
- random shocks

Each training episode uses a different seed, so the agent cannot simply memorize one price series.

The environment exposes 12 normalized market/portfolio observations. V6 does not feed this dense vector directly into the recurrent circuit. Instead it uses a compact sparse state representation inspired by the principle of sparse coding observed in the Drosophila mushroom body: only the four strongest feature responses remain active, represented through positive/negative channels. A three-unit previous-action context is added without introducing a large memory model.

The resulting V6 circuit is:

```text
12 market features
       |
       v
Sparse k-WTA encoder
27 state units / 5 active
       |
       v
32-neuron sparse recurrent circuit
       |
       v
TD(lambda) + eligibility traces
       |
       v
Opportunity gate
       |
       +---- BUY
       +---- SELL
       +---- HOLD
```

The opportunity gate keeps `HOLD` as a no-trade zone when neither `BUY` nor `SELL` has enough score advantage over `HOLD`. This is a small decision rule around the learned output circuit, not another neural network.

V6 also preserves the recurrent decision state correctly during TD updates, so when the next observation advances the circuit, the reward is still assigned to the state that produced the previous action.

The agent continues to use:

- sparse recurrent connections
- 32 hidden neurons
- 10% connection density
- TD learning
- eligibility traces
- transaction costs and drawdown-aware reward
- deterministic seeds
- multi-market unseen evaluation
- zero runtime ML dependencies

### Run the experiment

From the repository root:

```bash
python -m flydeck.cli
```

The CLI trains across 100 different synthetic markets and then evaluates the trained agent on a completely unseen market seed plus a 20-market evaluation set.

### Run tests

```bash
python -m pytest
```

No NumPy, PyTorch, exchange API, or other ML framework is required.

## Why the sparse representation

The Drosophila mushroom body is a useful biological reference for this stage because sensory information is represented by sparse populations of Kenyon cells, and sparse/decorrelated activity is associated with learned discrimination. The goal here is **not** to claim that this simple encoder reproduces the fly brain. It is a deliberately cheap computational abstraction that can be measured before any real connectome data is introduced.

The longer-term architecture remains:

```text
Sparse sensory/state representation
              |
              v
       Recurrent sparse circuit
              |
              v
       Eligibility / learning
              |
              v
          Action output
```

Later, the hand-designed sparse circuit can be compared against structures extracted from the MaleCNS connectome.

## Development direction

The intended progression is:

1. Build a functional and economical agent.
2. Stabilize sequential learning and out-of-sample behavior.
3. Measure performance against simple baselines across many unseen markets.
4. Add historical BTC/USDT OHLCV loaded from local files.
5. Expand to multiple crypto assets and more difficult market conditions.
6. Introduce a neuron/connection budget for controlled architecture comparisons.
7. Add the MaleCNS connectome as an architectural resource rather than immediately simulating every biological detail.
8. Map useful connectome structure into the efficient agent runtime.

The connectome should be introduced only after the current runtime has demonstrated that it can solve meaningful sequential tasks.

## Principles

- Functional before complex
- Low memory usage
- Sparse computation where possible
- Small dependency footprint
- Deterministic and inspectable internals
- Easy to measure
- Replaceable architecture

## Roadmap

- [x] Project repository
- [x] Minimal agent loop
- [x] Internal state / short-term memory
- [x] Lightweight learning mechanism
- [x] Environment interface
- [x] Sparse neural architecture
- [x] 2D navigation validation environment
- [x] Synthetic crypto market generator
- [x] Financial trading environment
- [x] Training across different market seeds
- [x] Sparse market state encoder
- [x] Opportunity gate for finance decisions
- [x] TD learning with eligibility traces
- [ ] Resource usage metrics
- [ ] Historical OHLCV loader
- [ ] Multi-asset environment
- [ ] Neuron/connection budget experiments
- [ ] Connectome data adapter
- [ ] Connectome-derived agent architecture
- [ ] Tool/action interface
