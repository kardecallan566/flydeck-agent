# FlyDeck Agent

A lightweight agent built with efficiency as a first-class goal.

The long-term goal is to explore whether structural principles from the *Drosophila* connectome can help build capable agents with low computational and memory requirements.

## Current direction

The project starts **without the fly connectome**. The first milestone is a small, functional agent runtime that can perceive state, maintain internal memory, select actions, and learn from feedback while keeping computation and dependencies minimal.

The connectome-inspired architecture will be introduced after the agent runtime is working.

## Principles

- Functional before complex
- Low memory usage
- Sparse computation where possible
- Small dependency footprint
- Deterministic and inspectable internals
- Easy to measure
- Replaceable architecture

## Navigation validation

The first behavioral validation task is a small 2D grid world. The agent receives six values:

1. blocked/open flag for up
2. blocked/open flag for right
3. blocked/open flag for down
4. blocked/open flag for left
5. normalized horizontal direction to the goal
6. normalized vertical direction to the goal

It can choose four actions: up, right, down, or left. Reaching the goal gives a strong positive reward; wasting steps, moving away, or hitting obstacles produces negative feedback.

The environment is deterministic so the same run can be reproduced locally.

### Run the training demo

From the repository root:

```bash
python -m flydeck.cli
```

The demo trains for 500 episodes and then runs one greedy evaluation episode. It prints the map, connection count, rewards, success count, and memory size.

### Run tests

```bash
python -m pytest
```

No NumPy, PyTorch, or other ML framework is required.

## Roadmap

- [x] Project repository
- [x] Minimal agent loop
- [x] Internal state / short-term memory
- [x] Lightweight learning mechanism
- [x] Environment interface
- [x] Sparse neural architecture
- [x] 2D navigation validation environment
- [ ] Resource usage metrics
- [ ] Connectome data adapter
- [ ] Connectome-derived agent architecture
- [ ] Tool/action interface
