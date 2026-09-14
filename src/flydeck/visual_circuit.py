from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .malecns import NT_SIGN

T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")
ON_PATHWAY_TYPES = ("L1", "Mi1", "Mi4", "Mi9", "Tm3", "C3", "TmY15", "CT1")
OFF_PATHWAY_TYPES = ("L2", "Tm1", "Tm2", "Tm4", "Tm9", "CT1")
MOTION_TYPES = tuple(dict.fromkeys(ON_PATHWAY_TYPES + OFF_PATHWAY_TYPES + T4_TYPES + T5_TYPES))


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
    l1_inputs: tuple[int, ...]
    l2_inputs: tuple[int, ...]
    t4_outputs: tuple[tuple[int, ...], ...]
    t5_outputs: tuple[tuple[int, ...], ...]

    @property
    def on_inputs(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view: L1 is the ON entry channel."""
        return (self.l1_inputs, (), (), ())

    @property
    def off_inputs(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view: L2 is the OFF entry channel."""
        return (self.l2_inputs, (), (), ())

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": 2,
            "source": "MaleCNS v1.0",
            "architecture": "L1/L2 -> medulla motion pathway -> T4/T5",
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
            "l1_inputs": list(self.l1_inputs),
            "l2_inputs": list(self.l2_inputs),
            "t4_outputs": [list(group) for group in self.t4_outputs],
            "t5_outputs": [list(group) for group in self.t5_outputs],
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VisualCircuit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        neurons = tuple(
            VisualNeuron(
                int(item["body_id"]),
                str(item["cell_type"]),
                str(item["role"]),
                item.get("neurotransmitter"),
                float(item.get("sign", 0.0)),
            )
            for item in payload["neurons"]
        )
        if "l1_inputs" in payload:
            l1 = tuple(int(i) for i in payload["l1_inputs"])
            l2 = tuple(int(i) for i in payload["l2_inputs"])
        else:
            # Read the original experimental schema without treating its four
            # artificial pools as biological directional channels.
            l1 = tuple(int(i) for i in payload.get("on_inputs", [[]])[0])
            l2 = tuple(int(i) for i in payload.get("off_inputs", [[]])[0])
        return cls(
            neurons=neurons,
            edges=tuple(VisualEdge(int(e[0]), int(e[1]), float(e[2])) for e in payload["edges"]),
            l1_inputs=l1,
            l2_inputs=l2,
            t4_outputs=tuple(tuple(int(i) for i in group) for group in payload["t4_outputs"]),
            t5_outputs=tuple(tuple(int(i) for i in group) for group in payload["t5_outputs"]),
        )


class VisualCircuitBuilder:
    """Extract an anatomically named motion pathway from the published graph.

    We never invent a connection. Every retained edge is a row from the
    MaleCNS connection-weight table and every retained neuron is selected by
    its published cell type annotation. The compact circuit contains the named
    L1/L2 -> T4/T5 motion pathway and its major documented interneurons.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        weights_path: str | Path,
        neurotransmitters_path: str | Path | None = None,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.weights_path = Path(weights_path)
        self.neurotransmitters_path = Path(neurotransmitters_path) if neurotransmitters_path else None

    def build(self, output_path: str | Path) -> VisualCircuit:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc

        annotations = feather.read_table(self.annotations_path)
        cols = set(annotations.column_names)
        body_col = _first(cols, "bodyId", "bodyid", "body_id", "id")
        type_col = _first(cols, "type", "cell_type", "cellType")
        status_col = "status" if "status" in cols else None
        data = annotations.select([body_col, type_col] + ([status_col] if status_col else [])).to_pydict()
        wanted = {name.lower() for name in MOTION_TYPES}
        selected: dict[int, str] = {}
        for i, body in enumerate(data[body_col]):
            if status_col and str(data[status_col][i]).lower() != "traced":
                continue
            cell_type = "" if data[type_col][i] is None else str(data[type_col][i]).strip()
            if cell_type.lower() in wanted:
                selected[int(body)] = cell_type
        if not selected:
            raise ValueError("No named MaleCNS visual motion types were found")

        ids = set(selected)
        value_set = pa.array(sorted(ids), type=pa.int64())
        edges_by_body: list[tuple[int, int, float]] = []
        with ipc.open_file(self.weights_path) as reader:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                mask = pc.and_(
                    pc.is_in(batch["body_pre"], value_set=value_set),
                    pc.is_in(batch["body_post"], value_set=value_set),
                )
                filtered = batch.filter(mask)
                edges_by_body.extend(
                    (int(source), int(target), float(weight))
                    for source, target, weight in zip(
                        filtered["body_pre"], filtered["body_post"], filtered["weight"]
                    )
                    if int(source) != int(target)
                )

        nt = self._neurotransmitters()
        ordered = sorted(ids)
        index = {body: i for i, body in enumerate(ordered)}
        neurons = tuple(self._make_neuron(body, selected[body], nt) for body in ordered)
        edges = tuple(VisualEdge(index[source], index[target], weight) for source, target, weight in edges_by_body)

        def group(name: str) -> tuple[int, ...]:
            return tuple(index[body] for body in ordered if selected[body].lower() == name.lower())

        circuit = VisualCircuit(
            neurons=neurons,
            edges=edges,
            l1_inputs=group("L1"),
            l2_inputs=group("L2"),
            t4_outputs=tuple(group(name) for name in T4_TYPES),
            t5_outputs=tuple(group(name) for name in T5_TYPES),
        )
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("MaleCNS circuit is missing L1 or L2 visual entry neurons")
        if any(not group for group in circuit.t4_outputs + circuit.t5_outputs):
            raise ValueError("MaleCNS circuit is missing one or more T4/T5 directional types")
        circuit.save(output_path)
        return circuit

    def _neurotransmitters(self) -> dict[int, str]:
        if not self.neurotransmitters_path:
            return {}
        import pyarrow.feather as feather
        table = feather.read_table(self.neurotransmitters_path)
        cols = set(table.column_names)
        body_col = _first(cols, "bodyId", "bodyid", "body_id", "id")
        nt_col = _first(cols, "predicted_nt", "neurotransmitter", "consensus_nt", "nt")
        data = table.select([body_col, nt_col]).to_pydict()
        return {
            int(body): str(value).strip()
            for body, value in zip(data[body_col], data[nt_col])
            if value is not None
        }

    @staticmethod
    def _make_neuron(body: int, cell_type: str, nt: dict[int, str]) -> VisualNeuron:
        low = cell_type.lower()
        if low in {name.lower() for name in T4_TYPES + T5_TYPES}:
            role = "motion_detector"
        elif low in {"l1", "l2"}:
            role = "visual_entry"
        else:
            role = "motion_interneuron"
        transmitter = nt.get(body)
        return VisualNeuron(body, cell_type, role, transmitter, NT_SIGN.get((transmitter or "").lower(), 0.0))


def _first(columns: set[str], *names: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Missing MaleCNS column; available={sorted(columns)}")
