from __future__ import annotations

from collections import Counter
from pathlib import Path

from .malecns import NT_SIGN, MaleCNSCircuit, MaleCNSEdge, MaleCNSNeuron


def build_degree_core_circuit(
    annotations_path: str | Path,
    weights_path: str | Path,
    output_path: str | Path,
    *,
    neuron_count: int = 2048,
    min_synapses: int = 3,
    feature_count: int = 12,
    action_count: int = 3,
    pool_size: int = 8,
) -> MaleCNSCircuit:
    """Extract a reproducible bounded core from the official MaleCNS tables.

    The raw graph contains far more neurons/edges than an online Python
    simulation should update every market tick. We therefore select the highest
    weighted-degree traced neurons, keep only edges inside that set, and expose
    deterministic input/output pools. This is a structural baseline, not a
    claim that these pools correspond to biological sensory or motor pathways.
    """
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise RuntimeError("Install the connectome extra with: pip install -e '.[connectome]'") from exc

    annotations = feather.read_table(annotations_path).to_pydict()
    ids = annotations.get("bodyId") or annotations.get("body_id")
    statuses = annotations.get("status")
    if ids is None:
        raise ValueError("annotation table must contain bodyId")

    traced = {
        int(body_id)
        for index, body_id in enumerate(ids)
        if statuses is None or str(statuses[index]).lower() == "traced"
    }
    if not traced:
        raise ValueError("no traced neurons found in annotations")

    weights = feather.read_table(weights_path).to_pydict()
    pre = weights.get("body_pre")
    post = weights.get("body_post")
    weight = weights.get("weight")
    if pre is None or post is None or weight is None:
        raise ValueError("weights table must contain body_pre, body_post and weight")

    degree: Counter[int] = Counter()
    raw_edges: list[tuple[int, int, float]] = []
    for source, target, count in zip(pre, post, weight):
        source = int(source)
        target = int(target)
        count = int(count)
        if count < min_synapses or source not in traced or target not in traced:
            continue
        raw_edges.append((source, target, float(count)))
        degree[source] += count
        degree[target] += count

    selected_ids = [body_id for body_id, _ in degree.most_common(neuron_count)]
    if len(selected_ids) < max(feature_count * pool_size, action_count * pool_size):
        raise ValueError("selected MaleCNS core is too small for requested pools")
    selected = set(selected_ids)
    ordered_ids = sorted(selected_ids)
    index = {body_id: position for position, body_id in enumerate(ordered_ids)}

    edges = tuple(
        MaleCNSEdge(index[source], index[target], _signed_weight(count))
        for source, target, count in raw_edges
        if source in selected and target in selected
    )

    input_neurons = tuple(
        tuple(range(offset, offset + pool_size))
        for offset in range(0, feature_count * pool_size, pool_size)
    )
    output_start = feature_count * pool_size
    output_neurons = tuple(
        tuple(range(output_start + offset, output_start + offset + pool_size))
        for offset in range(0, action_count * pool_size, pool_size)
    )

    neurons = tuple(MaleCNSNeuron(body_id=body_id) for body_id in ordered_ids)
    circuit = MaleCNSCircuit(neurons, edges, input_neurons, output_neurons)
    circuit.save(output_path)
    return circuit


def _signed_weight(count: float) -> float:
    """V1 sign-neutral fallback; neurotransmitter-aware signing comes next."""
    return count / (1.0 + count)
