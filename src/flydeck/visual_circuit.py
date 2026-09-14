from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .malecns import NT_SIGN

T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")
ON_TYPES = ("L1", "Mi1", "Mi4", "Mi9", "Tm3", "TmY15", "CT1", "C3")
OFF_TYPES = ("L2", "Tm1", "Tm2", "Tm4", "Tm9", "TmY15", "CT1")
MOTION_TYPES = tuple(dict.fromkeys(T4_TYPES + T5_TYPES + ON_TYPES + OFF_TYPES))


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
            "architecture": "named L1/L2 -> T4/T5 motion pathway",
            "neurons": [n.__dict__ if hasattr(n, "__dict__") else {"body_id": n.body_id, "cell_type": n.cell_type, "role": n.role, "neurotransmitter": n.neurotransmitter, "sign": n.sign} for n in self.neurons],
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
        return cls(
            neurons=tuple(VisualNeuron(int(n["body_id"]), str(n["cell_type"]), str(n["role"]), n.get("neurotransmitter"), float(n.get("sign", 0))) for n in payload["neurons"]),
            edges=tuple(VisualEdge(int(e[0]), int(e[1]), float(e[2])) for e in payload["edges"]),
            on_inputs=tuple(tuple(g) for g in payload["on_inputs"]),
            off_inputs=tuple(tuple(g) for g in payload["off_inputs"]),
            t4_outputs=tuple(tuple(g) for g in payload["t4_outputs"]),
            t5_outputs=tuple(tuple(g) for g in payload["t5_outputs"]),
        )


class VisualCircuitBuilder:
    """Extract named, functionally documented motion-vision cell types.

    This deliberately replaces the old degree-core selection. L1/L2 provide
    ON/OFF entry, Mi/Tm/CT1/C3/TmY15 are known motion-pathway intermediates,
    and T4/T5 are the first direction-selective neurons.
    """

    def __init__(self, annotations_path: str | Path, weights_path: str | Path, neurotransmitters_path: str | Path | None = None) -> None:
        self.annotations_path = Path(annotations_path)
        self.weights_path = Path(weights_path)
        self.neurotransmitters_path = Path(neurotransmitters_path) if neurotransmitters_path else None

    def build(self, output_path: str | Path) -> VisualCircuit:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc

        table = feather.read_table(self.annotations_path)
        cols = set(table.column_names)
        body_col = _first(cols, "bodyId", "body_id", "id")
        type_col = _first(cols, "type", "cell_type", "cellType")
        status_col = "status" if "status" in cols else None
        wanted = {x.lower() for x in MOTION_TYPES}
        selected: dict[int, str] = {}
        data = table.select([body_col, type_col] + ([status_col] if status_col else [])).to_pydict()
        for i, body in enumerate(data[body_col]):
            if status_col and str(data[status_col][i]).lower() != "traced":
                continue
            value = "" if data[type_col][i] is None else str(data[type_col][i])
            if value.lower() in wanted:
                selected[int(body)] = value
        if not selected:
            raise ValueError("No named visual motion neuron types were found")

        ids = set(selected)
        edges: list[tuple[int, int, float]] = []
        value_set = pa.array(list(ids), type=pa.int64())
        with ipc.open_file(self.weights_path) as reader:
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                mask = pc.and_(pc.is_in(batch["body_pre"], value_set=value_set), pc.is_in(batch["body_post"], value_set=value_set))
                batch = batch.filter(mask)
                edges.extend((int(a), int(b), float(w)) for a, b, w in zip(batch["body_pre"], batch["body_post"], batch["weight"]) if int(a) != int(b))

        ordered = sorted(ids)
        index = {body: i for i, body in enumerate(ordered)}
        neurons = tuple(self._neuron(body, selected[body]) for body in ordered)
        compact_edges = tuple(VisualEdge(index[a], index[b], w) for a, b, w in edges)
        on = self._partition(selected, index, ON_TYPES)
        off = self._partition(selected, index, OFF_TYPES)
        t4 = tuple(self._group(selected, index, name) for name in T4_TYPES)
        t5 = tuple(self._group(selected, index, name) for name in T5_TYPES)
        circuit = VisualCircuit(neurons, compact_edges, on, off, t4, t5)
        circuit.save(output_path)
        return circuit

    def _neuron(self, body: int, cell_type: str) -> VisualNeuron:
        nt = self._neurotransmitters().get(body)
        low = cell_type.lower()
        role = "motion_detector" if low in {x.lower() for x in T4_TYPES + T5_TYPES} else "visual_entry" if low in {"l1", "l2"} else "visual_interneuron"
        return VisualNeuron(body, cell_type, role, nt, NT_SIGN.get((nt or "").lower(), 0.0))

    def _neurotransmitters(self) -> dict[int, str]:
        if not self.neurotransmitters_path:
            return {}
        import pyarrow.feather as feather
        table = feather.read_table(self.neurotransmitters_path)
        cols = set(table.column_names)
        body_col = _first(cols, "bodyId", "body_id", "id")
        nt_col = _first(cols, "predicted_nt", "neurotransmitter", "consensus_nt")
        data = table.select([body_col, nt_col]).to_pydict()
        return {int(body): str(nt) for body, nt in zip(data[body_col], data[nt_col]) if nt is not None}

    @staticmethod
    def _group(annotations: dict[int, str], index: dict[int, int], name: str) -> tuple[int, ...]:
        return tuple(index[body] for body in sorted(annotations) if annotations[body].lower() == name.lower())

    @staticmethod
    def _partition(annotations: dict[int, str], index: dict[int, int], names: tuple[str, ...], parts: int = 4) -> tuple[tuple[int, ...], ...]:
        wanted = {name.lower() for name in names}
        ids = [index[body] for body in sorted(annotations) if annotations[body].lower() in wanted]
        groups = [[] for _ in range(parts)]
        for position, neuron in enumerate(ids):
            groups[position % parts].append(neuron)
        return tuple(tuple(group) for group in groups)


def _first(columns: set[str], *names: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Missing required MaleCNS column; available={sorted(columns)}")
