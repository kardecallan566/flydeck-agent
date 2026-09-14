from __future__ import annotations

from dataclasses import dataclass

from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class ReceptiveField:
    x: float
    y: float
    excitatory_x: float | None
    excitatory_y: float | None
    inhibitory_x: float | None
    inhibitory_y: float | None
    excitatory_mass: float
    inhibitory_mass: float

    @property
    def has_inhibitory_component(self) -> bool:
        return self.inhibitory_x is not None and self.inhibitory_y is not None

    @property
    def excitation_inhibition_offset(self) -> tuple[float, float] | None:
        if not self.has_inhibitory_component or self.excitatory_x is None or self.excitatory_y is None:
            return None
        return (
            self.inhibitory_x - self.excitatory_x,
            self.inhibitory_y - self.excitatory_y,
        )


def infer_receptive_fields(
    circuit: VisualCircuit,
    *,
    iterations: int = 12,
) -> dict[int, ReceptiveField]:
    """Propagate L1/L2 spatial coordinates through the selected circuit.

    Absolute synaptic weight is used for spatial influence, while the source
    neurotransmitter sign is retained separately so excitatory and inhibitory
    receptive-field centers can be compared. This is an anatomical influence
    estimate, not a claim that soma coordinates are a true eye retinotopy.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")

    entry_ids = set(circuit.l1_inputs) | set(circuit.l2_inputs)
    positions: dict[int, tuple[float, float]] = {}
    for index in entry_ids:
        neuron = circuit.neurons[index]
        if neuron.spatial_x is not None and neuron.spatial_y is not None:
            positions[index] = (neuron.spatial_x, neuron.spatial_y)

    incoming: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]
    for edge in circuit.edges:
        if abs(edge.weight) > 1e-12:
            incoming[edge.target].append((edge.source, abs(edge.weight)))

    for _ in range(iterations):
        previous = positions.copy()
        for target, sources in enumerate(incoming):
            if target in entry_ids:
                continue
            weighted = [
                (source, weight)
                for source, weight in sources
                if source in previous
            ]
            if not weighted:
                continue
            total = sum(weight for _, weight in weighted)
            if total <= 1e-12:
                continue
            positions[target] = (
                sum(previous[source][0] * weight for source, weight in weighted) / total,
                sum(previous[source][1] * weight for source, weight in weighted) / total,
            )

    fields: dict[int, ReceptiveField] = {}
    for index, (x, y) in positions.items():
        excitatory = _component_center(circuit, index, positions, positive=True)
        inhibitory = _component_center(circuit, index, positions, positive=False)
        fields[index] = ReceptiveField(
            x=x,
            y=y,
            excitatory_x=excitatory[0] if excitatory else None,
            excitatory_y=excitatory[1] if excitatory else None,
            inhibitory_x=inhibitory[0] if inhibitory else None,
            inhibitory_y=inhibitory[1] if inhibitory else None,
            excitatory_mass=excitatory[2] if excitatory else 0.0,
            inhibitory_mass=inhibitory[2] if inhibitory else 0.0,
        )
    return fields


def _component_center(
    circuit: VisualCircuit,
    target: int,
    positions: dict[int, tuple[float, float]],
    *,
    positive: bool,
) -> tuple[float, float, float] | None:
    weighted: list[tuple[float, float, float]] = []
    for edge in circuit.edges:
        if edge.target != target or edge.source not in positions:
            continue
        sign = circuit.neurons[edge.source].sign
        if (sign >= 0.0) != positive:
            continue
        weight = abs(edge.weight)
        if weight <= 1e-12:
            continue
        x, y = positions[edge.source]
        weighted.append((x, y, weight))
    if not weighted:
        return None
    total = sum(weight for _, _, weight in weighted)
    return (
        sum(x * weight for x, _, weight in weighted) / total,
        sum(y * weight for _, y, weight in weighted) / total,
        total,
    )


def summarize_receptive_fields(
    circuit: VisualCircuit,
    fields: dict[int, ReceptiveField],
) -> str:
    """Return a compact diagnostic summary for T4/T5 spatial organization."""
    lines = []
    for label, groups in (("T4", circuit.t4_outputs), ("T5", circuit.t5_outputs)):
        for subtype, group in zip("abcd", groups):
            values = [fields[index] for index in group if index in fields]
            if not values:
                lines.append(f"{label}{subtype}: no receptive field")
                continue
            x = sum(value.x for value in values) / len(values)
            y = sum(value.y for value in values) / len(values)
            offsets = [value.excitation_inhibition_offset for value in values]
            offsets = [offset for offset in offsets if offset is not None]
            if offsets:
                ox = sum(offset[0] for offset in offsets) / len(offsets)
                oy = sum(offset[1] for offset in offsets) / len(offsets)
                offset_text = f" EI_offset=({ox:+.4f},{oy:+.4f})"
            else:
                offset_text = " EI_offset=(none)"
            lines.append(f"{label}{subtype}: rf=({x:.4f},{y:.4f}){offset_text}")
    return "\n".join(lines)
