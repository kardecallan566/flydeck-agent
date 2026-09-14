from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

from .malecns import NT_SIGN, MALECNS_BASE_URL

ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"

T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")
ENTRY_TYPES = ("L1", "L2")


@dataclass(frozen=True, slots=True)
class VisualNeuron:
    body_id: int
    cell_type: str
    role: str
    neurotransmitter: str | None
    sign: float


@dataclass(frozen=True, slots=True)
class VisualEdge:
    source: int
    target: int
    weight: float


@dataclass(frozen=True, slots=True)
class VisualCircuit:
    """An anatomically selected MaleCNS visual motion circuit."""

    neurons: tuple[VisualNeuron, ...]
    edges: tuple[VisualEdge, ...]
    on_inputs: tuple[tuple[int, ...], ...]
    off_inputs: tuple[tuple[int, ...], ...]
    t4_outputs: tuple[tuple[int, ...], ...]
    t5_outputs: tuple[tuple[int, ...], ...]

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "source": "MaleCNS v1.0",
            "architecture": "L1/L2 -> T4/T5 visual motion pathway",
            "neurons": [
                {
                    "body_id": n.body_id,
                    "cell_type": n.cell_type,
                    "role": n.role,
                    "neurotransmitter": n.neurotransmitter,
                    "sign": n.sign,
                }
                for n in self.neurons
            ],
            "edges": [[e.source, e.target, e.weight] for e in self.edges],
            "on_inputs": [list(g) for g in self.on_inputs],
            "off_inputs": [list(g) for g in self.off_inputs],
            "t4_outputs": [list(g) for g in self.t4_outputs],
            "t5_outputs": [list(g) for g in self.t5_outputs],
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VisualCircuit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        neurons = tuple(
            VisualNeuron(
                body_id=int(n["body_id"]),
                cell_type=str(n["cell_type"]),
                role=str(n["role"]),
                neurotransmitter=n.get("neurotransmitter"),
                sign=float(n.get("sign", 0.0)),
            )
            for n in payload["neurons"]
        )
        edges = tuple(VisualEdge(int(e[0]), int(e[1]), float(e[2])) for e in payload["edges"])
        return cls(
            neurons=neurons,
            edges=edges,
            on_inputs=tuple(tuple(g) for g in payload["on_inputs"]),
            off_inputs=tuple(tuple(g) for g in payload["off_inputs"]),
            t4_outputs=tuple(tuple(g) for g in payload["t4_outputs"]),
            t5_outputs=tuple(tuple(g) for g in payload["t5_outputs"]),
        )


class VisualCircuitBuilder:
    """Extract only neurons connected to the identified L1/L2/T4/T5 pathway.

    Selection is based on annotated cell type plus published MaleCNS connectivity;
    no neuron is selected because it merely has a high degree. Upstream expansion
    follows real directed edges toward L1/L2 and downstream expansion preserves
    the first integration layer after T4/T5.
    """

    def __init__(self, annotations_path: str | Path, weights_path: str | Path, neurotransmitters_path: str | Path | None = None) -> None:
        self.annotations_path = Path(annotations_path)
        self.weights_path = Path(weights_path)
        self.neurotransmitters_path = Path(neurotransmitters_path) if neurotransmitters_path else None

    def build(self, output_path: str | Path, upstream_depth: int = 2, downstream_depth: int = 1) -> VisualCircuit:
        try:
            import pyarrow.feather as feather
            import pyarrow.compute as pc
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError("Install the connectome extra with: pip install -e '.[connectome]'") from exc

        table = feather.read_table(self.annotations_path)
        columns = set(table.column_names)
        body_col = _first(columns, "bodyId", "body_id", "id")
        type_col = _first(columns, "type", "cell_type", "cellType")
        status_col = _first(columns, "status")
        data = table.select([body_col, type_col] + ([status_col] if status_col else [])).to_pydict()
        annotations: dict[int, str] = {}
        for i, body in enumerate(data[body_col]):
            if status_col and str(data[status_col][i]).lower() != "traced":
                continue
            cell_type = str(data[type_col][i]) if data[type_col][i] is not None else ""
            if cell_type and cell_type.lower() != "nan":
                annotations[int(body)] = cell_type

        def ids_for(types: tuple[str, ...]) -> set[int]:
            wanted = {value.lower() for value in types}
            return {body for body, cell_type in annotations.items() if cell_type.lower() in wanted}

        t4_ids = ids_for(T4_TYPES)
        t5_ids = ids_for(T5_TYPES)
        l1_ids = ids_for(("L1",))
        l2_ids = ids_for(("L2",))
        if not t4_ids or not t5_ids:
            raise ValueError("MaleCNS annotations did not contain T4a-d and T5a-d cell types")
        if not l1_ids or not l2_ids:
            raise ValueError("MaleCNS annotations did not contain L1 and L2 cell types")

        selected = set(t4_ids) | set(t5_ids) | set(l1_ids) | set(l2_ids)
        retained_edges: dict[tuple[int, int], int] = {}
        frontier = set(t4_ids) | set(t5_ids)

        # Expand upstream from T4/T5 through real pre->post connectivity. The
        # L1/L2 populations are always retained as the visual entry points.
        for _ in range(upstream_depth):
            incoming = self._edges_touching(frontier, direction="incoming", ipc=ipc, pc=pc)
            parents = {source for source, _target, _weight in incoming}
            selected.update(parents)
            frontier = parents - selected.intersection(frontier)
            for source, target, weight in incoming:
                retained_edges[(source, target)] = weight
            if not frontier:
                break

        # Keep direct L1/L2-to-selected and selected-to-next-layer edges even
        # when they were not reached by the bounded expansion above.
        entry_frontier = l1_ids | l2_ids
        incoming_to_targets = self._edges_touching(selected, direction="incoming", ipc=ipc, pc=pc, allowed_sources=entry_frontier)
        for source, target, weight in incoming_to_targets:
            selected.add(source)
            retained_edges[(source, target)] = weight

        frontier = t4_ids | t5_ids
        for _ in range(downstream_depth):
            outgoing = self._edges_touching(frontier, direction="outgoing", ipc=ipc, pc=pc)
            children = {target for _source, target, _weight in outgoing}
            selected.update(children)
            frontier = children
            for source, target, weight in outgoing:
                retained_edges[(source, target)] = weight
            if not frontier:
                break

        # The previous targeted scans intentionally avoid loading the full graph
        # into memory. A final pass captures all edges whose endpoints survived.
        final_edges = self._edges_between(selected, ipc=ipc, pc=pc)
        for source, target, weight in final_edges:
            retained_edges[(source, target)] = weight

        ordered = sorted(selected)
        index = {body: i for i, body in enumerate(ordered)}
        neurons = self._make_neurons(ordered, annotations)
        edges = tuple(
            VisualEdge(index[source], index[target], weight)
            for (source, target), weight in retained_edges.items()
            if source in index and target in index and source != target
        )

        on = self._partition(l1_ids & selected, index)
        off = self._partition(l2_ids & selected, index)
        t4 = tuple(tuple(index[body] for body in sorted(t4_ids & selected)) for _ in [0])
        t5 = tuple(tuple(index[body] for body in sorted(t5_ids & selected)) for _ in [0])

        # Directional T4/T5 outputs are represented by the annotated a/b/c/d types.
        t4 = tuple(tuple(index[body] for body in sorted(t4_ids & selected) if annotations[body].lower() == cell.lower()) for cell in T4_TYPES)
        t5 = tuple(tuple(index[body] for body in sorted(t5_ids & selected) if annotations[body].lower() == cell.lower()) for cell in T5_TYPES)
        circuit = VisualCircuit(neurons, edges, on, off, t4, t5)
        circuit.save(output_path)
        return circuit

    def _edges_touching(self, node_ids: set[int], *, direction: str, ipc, pc, allowed_sources: set[int] | None = None) -> list[tuple[int, int, float]]:
        result: list[tuple[int, int, float]] = []
        with ipc.open_file(self.weights_path) as reader:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                source_col = batch.column(batch.schema.get_field_index("body_pre"))
                target_col = batch.column(batch.schema.get_field_index("body_post"))
                if direction == "incoming":
                    mask = pc.is_in(target_col, value_set=_array(node_ids, pc, source_col))
                    if allowed_sources:
                        mask = pc.and_(mask, pc.is_in(source_col, value_set=_array(allowed_sources, pc, source_col)))
                else:
                    mask = pc.is_in(source_col, value_set=_array(node_ids, pc, source_col))
                filtered = batch.filter(mask)
                for source, target, weight in zip(filtered["body_pre"], filtered["body_post"], filtered["weight"]):
                    result.append((int(source), int(target), float(weight)))
        return result

    def _edges_between(self, node_ids: set[int], *, ipc, pc) -> list[tuple[int, int, float]]:
        result: list[tuple[int, int, float]] = []
        with ipc.open_file(self.weights_path) as reader:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                source_col = batch["body_pre"]
                target_col = batch["body_post"]
                source_mask = pc.is_in(source_col, value_set=_array(node_ids, pc, source_col))
                target_mask = pc.is_in(target_col, value_set=_array(node_ids, pc, target_col))
                filtered = batch.filter(pc.and_(source_mask, target_mask))
                for source, target, weight in zip(filtered["body_pre"], filtered["body_post"], filtered["weight"]):
                    result.append((int(source), int(target), float(weight)))
        return result

    def _make_neurons(self, ordered: list[int], annotations: dict[int, str]) -> tuple[VisualNeuron, ...]:
        nt = self._load_neurotransmitters(ordered)
        neurons = []
        for body in ordered:
            cell_type = annotations[body]
            lowered = cell_type.lower()
            if lowered in {x.lower() for x in T4_TYPES + T5_TYPES}:
                role = "motion_detector"
            elif lowered in {"l1", "l2"}:
                role = "visual_entry"
            else:
                role = "visual_interneuron"
            transmitter = nt.get(body)
            neurons.append(VisualNeuron(body, cell_type, role, transmitter, NT_SIGN.get((transmitter or "").lower(), 0.0)))
        return tuple(neurons)

    def _load_neurotransmitters(self, body_ids: list[int]) -> dict[int, str]:
        if not self.neurotransmitters_path:
            return {}
        try:
            import pyarrow.feather as feather
        except ImportError as exc:
            raise RuntimeError("Install the connectome extra with: pip install -e '.[connectome]'") from exc
        table = feather.read_table(self.neurotransmitters_path)
        body_col = _first(set(table.column_names), "bodyId", "body_id", "id")
        nt_col = _first(set(table.column_names), "predicted_nt", "neurotransmitter", "consensus_nt")
        wanted = set(body_ids)
        data = table.select([body_col, nt_col]).to_pydict()
        result: dict[int, str] = {}
        for body, value in zip(data[body_col], data[nt_col]):
            body = int(body)
            if body in wanted and value is not None:
                result[body] = str(value)
        return result

    @staticmethod
    def _partition(body_ids: set[int], index: dict[int, int], parts: int = 4) -> tuple[tuple[int, ...], ...]:
        ordered = sorted(index[body] for body in body_ids)
        groups = [[] for _ in range(parts)]
        for position, neuron in enumerate(ordered):
            groups[position % parts].append(neuron)
        return tuple(tuple(group) for group in groups)


def _first(columns: set[str], *names: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"required MaleCNS column not found; available columns: {sorted(columns)}")


def _array(values: set[int], pc, reference_column):
    import pyarrow as pa
    return pa.array(list(values), type=reference_column.type)
