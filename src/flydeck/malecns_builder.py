from __future__ import annotations

from collections import Counter
from pathlib import Path

from .malecns import MaleCNSCircuit, MaleCNSEdge, MaleCNSNeuron


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
    """Extract a bounded core from the official MaleCNS tables.

    The raw graph is much larger than an online Python simulation should update
    every market tick. The weights file is scanned in Arrow record batches so
    the full 1.1 GB table is never converted into a giant Python list.
    Input/output pools are deterministic structural probes, not claims about
    biological sensory or motor pathways.
    """
    try:
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("Install the connectome extra with: pip install -e '.[connectome]'") from exc

    annotations = feather.read_table(annotations_path, columns=["bodyId", "status"]).to_pydict()
    ids = annotations["bodyId"]
    statuses = annotations.get("status")
    traced = {
        int(body_id)
        for index, body_id in enumerate(ids)
        if statuses is None or str(statuses[index]).lower() == "traced"
    }
    if not traced:
        raise ValueError("no traced neurons found in annotations")

    degree: Counter[int] = Counter()
    with ipc.open_file(weights_path) as reader:
        for batch in reader.iter_batches(batch_size=1_000_000):
            columns = batch.to_pydict()
            for source, target, count in zip(
                columns["body_pre"], columns["body_post"], columns["weight"]
            ):
                source = int(source)
                target = int(target)
                count = int(count)
                if count < min_synapses or source not in traced or target not in traced:
                    continue
                degree[source] += count
                degree[target] += count

    selected_ids = [body_id for body_id, _ in degree.most_common(neuron_count)]
    required = feature_count * pool_size + action_count * pool_size
    if len(selected_ids) < required:
        raise ValueError("selected MaleCNS core is too small for requested pools")
    selected = set(selected_ids)
    ordered_ids = sorted(selected_ids)
    index = {body_id: position for position, body_id in enumerate(ordered_ids)}

    edges: list[MaleCNSEdge] = []
    with ipc.open_file(weights_path) as reader:
        for batch in reader.iter_batches(batch_size=1_000_000):
            columns = batch.to_pydict()
            for source, target, count in zip(
                columns["body_pre"], columns["body_post"], columns["weight"]
            ):
                source = int(source)
                target = int(target)
                count = int(count)
                if count < min_synapses or source not in selected or target not in selected:
                    continue
                edges.append(MaleCNSEdge(index[source], index[target], _normalized_weight(count)))

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
    circuit = MaleCNSCircuit(neurons, tuple(edges), input_neurons, output_neurons)
    circuit.save(output_path)
    return circuit


def _normalized_weight(count: int) -> float:
    """Compress synapse counts to a stable recurrent range without inventing sign."""
    return count / (1.0 + count)
