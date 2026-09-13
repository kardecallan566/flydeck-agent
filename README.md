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

## Roadmap

- [x] Project repository
- [ ] Minimal agent loop
- [ ] Internal state / short-term memory
- [ ] Lightweight learning mechanism
- [ ] Environment interface
- [ ] Resource usage metrics
- [ ] Sparse neural architecture
- [ ] Connectome data adapter
- [ ] Connectome-derived agent architecture
- [ ] Tool/action interface
