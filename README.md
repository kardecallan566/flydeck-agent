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

The agent receives 12 normalized observations containing recent returns, volatility, relative volume, price position, portfolio position, cash ratio, and current portfolio performance.

It chooses one of three actions:

- `HOLD`
- `BUY` — allocate another 25% of available cash
- `SELL` — sell 25% of the current asset position

A transaction fee is included. Reward is the percentage change in portfolio value after the next market step. The environment also tracks return, maximum drawdown, portfolio value, and trade count.

This makes the task useful for testing the current sparse recurrent architecture: the market is noisy, decisions are sequential, the useful signal changes with the regime, and the agent has to maintain internal state while managing its position.

### Run the experiment

From the repository root:

```bash
python -m flydeck.cli
```

The CLI trains across 100 different synthetic markets and then evaluates the trained agent on a completely unseen market seed.

### Run tests

```bash
python -m pytest
```

No NumPy, PyTorch, exchange API, or other ML framework is required.

## Development direction

The intended progression is:

1. Stress the lightweight agent on synthetic financial environments.
2. Measure behavior, memory, connection count, drawdown, and computational cost.
3. Add historical BTC/USDT OHLCV loaded from local files.
4. Expand to multiple crypto assets and more difficult market conditions.
5. Introduce a neuron/connection budget for controlled architecture comparisons.
6. Add the MaleCNS connectome as an architectural resource rather than immediately simulating every biological detail.
7. Map useful connectome structure into the efficient agent runtime.

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
- [ ] Resource usage metrics
- [ ] Historical OHLCV loader
- [ ] Multi-asset environment
- [ ] Neuron/connection budget experiments
- [ ] Connectome data adapter
- [ ] Connectome-derived agent architecture
- [ ] Tool/action interface
