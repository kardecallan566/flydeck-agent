from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

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
    spatial_x: float | None = None
    spatial_y: float | None = None


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
    spatial_mode: str = "none"

    @property
    def on_inputs(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view: L1 is the ON entry channel."""
        return (self.l1_inputs, (), (), ())

    @property
    def off_inputs(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view: L2 is the OFF entry channel."""
        return (self.l2_inputs, (), (), ())

    @property
    def has_spatial_mapping(self) -> bool:
        return self.spatial_mode != "none" and all(
            neuron.spatial_x is not None and neuron.spatial_y is not None
            for neuron in self.neurons
            if neuron.cell_type.lower() in {"l1", "l2"}
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": 3,
            "source": "MaleCNS v1.0",
            "architecture": "retinotopic L1/L2 -> medulla motion pathway -> T4/T5",
            "spatial_mode": self.spatial_mode,
            "neurons": [
                {
                    "body_id": n.body_id,
                    "cell_type": n.cell_type,
                    "role": n.role,
                    "neurotransmitter": n.neurotransmitter,
                    "sign": n.sign,
                    "spatial_x": n.spatial_x,
                    "spatial_y": n.spatial_y,
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
                _optional_float(item.get("spatial_x")),
                _optional_float(item.get("spatial_y")),
            )
            for item in payload["neurons"]
        )
        if "l1_inputs" in payload:
            l1 = tuple(int(i) for i in payload["l1_inputs"])
            l2 = tuple(int(i) for i in payload["l2_inputs"])
        else:
            l1 = tuple(int(i) for i in payload.get("on_inputs", [[]])[0])
            l2 = tuple(int(i) for i in payload.get("off_inputs", [[]])[0])
        return cls(
            neurons=neurons,
            edges=tuple(VisualEdge(int(e[0]), int(e[1]), float(e[2])) for e in payload["edges"]),
            l1_inputs=l1,
            l2_inputs=l2,
            t4_outputs=tuple(tuple(int(i) for i in group) for group in payload["t4_outputs"]),
            t5_outputs=tuple(tuple(int(i) for i in group) for group in payload["t5_outputs"]),
            spatial_mode=str(payload.get("spatial_mode", "none")),
        )


class VisualCircuitBuilder:
    """Extract a named MaleCNS motion pathway with an explicit spatial proxy.

    Neuron identities and directed edges come directly from MaleCNS v1.0.
    For L1/L2, the current retinotopic interface uses curated soma coordinates
    as a geometric proxy. This is intentionally labeled as a proxy: soma
    coordinates are not the published eye/medulla column map.
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
        spatial_cols = _find_spatial_columns(cols)
        selected_rows: list[tuple[int, str, float, float]] = []
        data = annotations.to_pylist()
        wanted = {name.lower() for name in MOTION_TYPES}
        for row in data:
            if status_col and str(row.get(status_col, "")).lower() != "traced":
                continue
            body = row.get(body_col)
            cell_type = "" if row.get(type_col) is None else str(row[type_col]).strip()
            if body is None or cell_type.lower() not in wanted:
                continue
            coords = _extract_soma_xyz(row, spatial_cols)
            if coords is None:
                if cell_type.lower() in {"l1", "l2"}:
                    raise ValueError(
                        "MaleCNS annotations do not expose parseable soma coordinates for L1/L2; "
                        "refusing to build a fake retinotopic map"
                    )
                continue
            selected_rows.append((int(body), cell_type, *coords))

        if not selected_rows:
            raise ValueError("No named MaleCNS visual motion types were found")

        ids = {body for body, *_ in selected_rows}
        raw_spatial = {body: (x, y, z) for body, _, x, y, z in selected_rows}
        selected = {body: cell_type for body, cell_type, *_ in selected_rows}
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

        l1_l2 = {body: raw_spatial[body] for body in ids if selected[body].lower() in {"l1", "l2"}}
        spatial_2d = _normalize_spatial(l1_l2)
        nt = self._neurotransmitters()
        ordered = sorted(ids)
        index = {body: i for i, body in enumerate(ordered)}
        neurons = tuple(
            self._make_neuron(body, selected[body], nt, spatial_2d.get(body))
            for body in ordered
        )
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
            spatial_mode="soma_xy_proxy",
        )
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("MaleCNS circuit is missing L1 or L2 visual entry neurons")
        if any(not group for group in circuit.t4_outputs + circuit.t5_outputs):
            raise ValueError("MaleCNS circuit is missing one or more T4/T5 directional types")
        if not circuit.has_spatial_mapping:
            raise ValueError("L1/L2 spatial mapping was not created")
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
    def _make_neuron(
        body: int,
        cell_type: str,
        nt: dict[int, str],
        spatial: tuple[float, float] | None,
    ) -> VisualNeuron:
        low = cell_type.lower()
        if low in {name.lower() for name in T4_TYPES + T5_TYPES}:
            role = "motion_detector"
        elif low in {"l1", "l2"}:
            role = "visual_entry"
        else:
            role = "motion_interneuron"
        transmitter = nt.get(body)
        return VisualNeuron(
            body,
            cell_type,
            role,
            transmitter,
            NT_SIGN.get((transmitter or "").lower(), 0.0),
            *(spatial or (None, None)),
        )


def _first(columns: set[str], *names: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Missing MaleCNS column; available={sorted(columns)}")


def _find_spatial_columns(columns: set[str]) -> tuple[str | None, str | None, str | None, str | None]:
    direct = (
        _first_optional(columns, "soma_x", "somaX", "soma_x_nm"),
        _first_optional(columns, "soma_y", "somaY", "soma_y_nm"),
        _first_optional(columns, "soma_z", "somaZ", "soma_z_nm"),
        _first_optional(columns, "somaLocation", "soma_location", "soma_position"),
    )
    if direct[0] and direct[1] and direct[2]:
        return direct
    return (None, None, None, direct[3])


def _extract_soma_xyz(
    row: dict,
    spatial_cols: tuple[str | None, str | None, str | None, str | None],
) -> tuple[float, float, float] | None:
    x_col, y_col, z_col, location_col = spatial_cols
    if x_col and y_col and z_col:
        try:
            return float(row[x_col]), float(row[y_col]), float(row[z_col])
        except (TypeError, ValueError):
            return None
    if not location_col:
        return None
    value = row.get(location_col)
    if isinstance(value, dict):
        keys = {str(k).lower(): v for k, v in value.items()}
        for names in (("x", "soma_x"), ("y", "soma_y"), ("z", "soma_z")):
            found = next((keys[name] for name in names if name in keys), None)
            if found is None:
                return None
            try:
                coords = tuple(float(keys[name]) for name in ("x", "y", "z"))
                return coords  # type: ignore[return-value]
            except (KeyError, TypeError, ValueError):
                return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
        if len(numbers) >= 3:
            return float(numbers[0]), float(numbers[1]), float(numbers[2])
    return None


def _normalize_spatial(coords: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float]]:
    if not coords:
        return {}
    xs = [value[0] for value in coords.values()]
    ys = [value[1] for value in coords.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    return {
        body: ((x - xmin) / xspan, (y - ymin) / yspan)
        for body, (x, y, _z) in coords.items()
    }


def _first_optional(columns: set[str], *names: str) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
