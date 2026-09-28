# MaleCNS Crypto V1

This experiment starts using the released MaleCNS v1.0 connectome as an actual computational substrate for the crypto agent.

## Goal

Test whether the topology of the fly nervous system provides a useful recurrent representation for market observations before introducing learned synaptic plasticity.

The first version deliberately separates:

- **biology:** frozen MaleCNS connectivity;
- **market interface:** 12 existing crypto observations;
- **learning:** only a tiny 3-action readout is trainable.

This lets us measure the value of the connectome itself instead of hiding it behind a conventional neural network.

## Dataset

The official MaleCNS v1.0 release contains the full male CNS connectome. The project provides neuron annotations, neurotransmitter predictions and a 1.1 GB segment-to-segment connection-weight table. The dataset is CC-BY. See the official download page before obtaining the raw files.

Raw data must stay outside Git.

## V1 circuit extraction

`malecns_builder.py` currently creates a bounded **degree-core** circuit:

1. keep annotated `Traced` neurons;
2. ignore edges below 3 synapses;
3. compute weighted in/out degree;
4. select the highest-degree neurons;
5. keep only edges inside the selected set;
6. normalize synapse counts to a stable positive recurrent range;
7. create deterministic input/output probe pools.

The input/output pools are explicitly a computational probe. They are **not** being presented as biological sensory or motor mappings.

The extractor scans the large Arrow table in batches instead of materializing the complete graph as Python objects.

## Important limitation

V1 does not yet use neurotransmitter signs. All retained edges are positive after normalization. This is intentional so the first experiment isolates topology. The next biological step is to sign connections using the released neurotransmitter predictions, with modulatory transmitters handled separately rather than pretending every transmitter is simply excitatory or inhibitory.

## Commands

Install optional tooling:

```bash
pip install -e ".[connectome]"
```

Download official inputs:

```bash
python -m flydeck.malecns_cli download --output data/malecns/raw
```

Build a 2,048-neuron degree core:

```bash
python -m flydeck.malecns_cli build \
  --annotations data/malecns/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --weights data/malecns/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --output data/malecns/degree_core_2048.json
```

Run the synthetic crypto experiment:

```bash
python -m flydeck.malecns_cli synthetic \
  --circuit data/malecns/degree_core_2048.json
```

Do not connect this experiment to live trading. Evaluation is offline only.
