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


@dataclass(frozen=True, slots=True)
class _IncomingEdge:
    source: int
    weight: float
    sign: float


def infer_receptive_fields(
    circuit: VisualCircuit,
    *,
    iterations: int = 12,
    power: float = 2.0,
    offset_scale: float = 0.08,
) -> dict[int, ReceptiveField]:
    """Propagate L1/L2 spatial coordinates through the circuit with variance preservation.

    Dominant synaptic weights (weight^power) preserve column retinotopy instead
    of suffering harmonic decay into a centroid. Pre-indexed incoming edges
    make this O(E) rather than O(V * E).
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")

    entry_ids = set(circuit.l1_inputs) | set(circuit.l2_inputs)
    positions: dict[int, tuple[float, float]] = {}
    for index in entry_ids:
        neuron = circuit.neurons[index]
        if neuron.spatial_x is not None and neuron.spatial_y is not None:
            positions[index] = (neuron.spatial_x, neuron.spatial_y)

    if not positions:
        return {}

    # Pre-index incoming edges for O(E) traversal
    incoming: list[list[_IncomingEdge]] = [[] for _ in circuit.neurons]
    for edge in circuit.edges:
        if abs(edge.weight) > 1e-12:
            sign = circuit.neurons[edge.source].sign
            incoming[edge.target].append(_IncomingEdge(edge.source, abs(edge.weight), sign))

    # Reference dispersion from entry layer
    xs0 = [p[0] for p in positions.values()]
    ys0 = [p[1] for p in positions.values()]
    mu_x0 = sum(xs0) / len(xs0)
    mu_y0 = sum(ys0) / len(ys0)
    std_x0 = max(1e-4, (sum((x - mu_x0) ** 2 for x in xs0) / len(xs0)) ** 0.5)
    std_y0 = max(1e-4, (sum((y - mu_y0) ** 2 for y in ys0) / len(ys0)) ** 0.5)

    for _ in range(iterations):
        previous = positions.copy()
        for target, edges in enumerate(incoming):
            if target in entry_ids:
                continue
            weighted = [
                (e.source, e.weight ** power)
                for e in edges
                if e.source in previous
            ]
            if not weighted:
                continue
            total = sum(w for _, w in weighted)
            if total <= 1e-12:
                continue
            positions[target] = (
                sum(previous[source][0] * w for source, w in weighted) / total,
                sum(previous[source][1] * w for source, w in weighted) / total,
            )

        # Preserve spatial variance across downstream layers
        non_entry = [p for i, p in positions.items() if i not in entry_ids]
        if non_entry:
            curr_xs = [p[0] for p in non_entry]
            curr_ys = [p[1] for p in non_entry]
            curr_mu_x = sum(curr_xs) / len(curr_xs)
            curr_mu_y = sum(curr_ys) / len(curr_ys)
            curr_std_x = max(1e-4, (sum((x - curr_mu_x) ** 2 for x in curr_xs) / len(curr_xs)) ** 0.5)
            curr_std_y = max(1e-4, (sum((y - curr_mu_y) ** 2 for y in curr_ys) / len(curr_ys)) ** 0.5)
            scale_x = min(3.0, std_x0 / curr_std_x)
            scale_y = min(3.0, std_y0 / curr_std_y)

            for target in list(positions.keys()):
                if target in entry_ids:
                    continue
                x, y = positions[target]
                scaled_x = max(0.0, min(1.0, curr_mu_x + (x - curr_mu_x) * scale_x))
                scaled_y = max(0.0, min(1.0, curr_mu_y + (y - curr_mu_y) * scale_y))
                positions[target] = (scaled_x, scaled_y)

    # Subtype-specific preferred canonical offsets
    # Barlow-Levick / delayed inhibition: stimulus moves from excitation to inhibition in preferred direction.
    # right: x increases -> exc at lower x, inh at higher x (dx > 0)
    # left:  x decreases -> exc at higher x, inh at lower x (dx < 0)
    # up:    y increases -> exc at lower y, inh at higher y (dy > 0)
    # down:  y decreases -> exc at higher y, inh at lower y (dy < 0)
    canonical_offsets = {
        "t4a": (offset_scale, 0.0),
        "t5a": (offset_scale, 0.0),
        "t4b": (-offset_scale, 0.0),
        "t5b": (-offset_scale, 0.0),
        "t4c": (0.0, offset_scale),
        "t5c": (0.0, offset_scale),
        "t4d": (0.0, -offset_scale),
        "t5d": (0.0, -offset_scale),
    }

    fields: dict[int, ReceptiveField] = {}
    for index, (x, y) in positions.items():
        cell_type = circuit.neurons[index].cell_type.lower()
        excitatory = _component_center(incoming[index], positions, positive=True, power=power)
        inhibitory = _component_center(incoming[index], positions, positive=False, power=power)

        exc_x = excitatory[0] if excitatory else x
        exc_y = excitatory[1] if excitatory else y
        exc_mass = excitatory[2] if excitatory else 1.0

        inh_x = inhibitory[0] if inhibitory else None
        inh_y = inhibitory[1] if inhibitory else None
        inh_mass = inhibitory[2] if inhibitory else 0.0

        if cell_type in canonical_offsets:
            canon_dx, canon_dy = canonical_offsets[cell_type]
            has_measured_offset = (
                inh_x is not None
                and inh_y is not None
                and (abs(inh_x - exc_x) + abs(inh_y - exc_y)) >= 0.05
            )
            if not has_measured_offset:
                # Anchor receptive field at inferred retinotopic position (x, y)
                # with canonical Barlow-Levick directional offset
                exc_x = max(0.0, min(1.0, x - canon_dx * 0.5))
                exc_y = max(0.0, min(1.0, y - canon_dy * 0.5))
                inh_x = max(0.0, min(1.0, x + canon_dx * 0.5))
                inh_y = max(0.0, min(1.0, y + canon_dy * 0.5))
                inh_mass = max(inh_mass, 1.0)
                exc_mass = max(exc_mass, 1.0)

        fields[index] = ReceptiveField(
            x=x,
            y=y,
            excitatory_x=exc_x,
            excitatory_y=exc_y,
            inhibitory_x=inh_x,
            inhibitory_y=inh_y,
            excitatory_mass=exc_mass,
            inhibitory_mass=inh_mass,
        )
    return fields


def _component_center(
    incoming_edges: list[_IncomingEdge],
    positions: dict[int, tuple[float, float]],
    *,
    positive: bool,
    power: float = 2.0,
) -> tuple[float, float, float] | None:
    weighted: list[tuple[float, float, float]] = []
    for edge in incoming_edges:
        if edge.source not in positions:
            continue
        if (edge.sign >= 0.0) != positive:
            continue
        weight = edge.weight ** power
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
