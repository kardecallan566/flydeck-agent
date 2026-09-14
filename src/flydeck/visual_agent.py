from __future__ import annotations

from dataclasses import dataclass
import math

from .market_retina import BNBMarketRetina, RetinaStimulus
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class VisualDecision:
    up_score: float
    down_score: float
    confidence: float
    wait: bool


class MaleCNSVisualSystem:
    """Run the compact, connectivity-preserving MaleCNS motion pathway."""

    def __init__(
        self,
        circuit: VisualCircuit,
        leak: float = 0.30,
        synapse_scale: float = 1.20,
        temporal_gain: float = 0.75,
        micro_steps: int = 4,
    ) -> None:
        if not 0.0 < leak <= 1.0 or synapse_scale <= 0.0 or temporal_gain < 0.0 or micro_steps < 1:
            raise ValueError("invalid visual dynamics")
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("visual circuit requires L1 and L2 entry neurons")
        if len(circuit.t4_outputs) != 4 or len(circuit.t5_outputs) != 4:
            raise ValueError("visual circuit requires four T4 and four T5 output groups")
        self.circuit = circuit
        self.leak = leak
        self.synapse_scale = synapse_scale
        self.temporal_gain = temporal_gain
        self.micro_steps = micro_steps
        self.state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]

        # Normalize each target's incoming synapse mass. This preserves the
        # relative MaleCNS connection weights without letting high-degree cells
        # dominate the rate dynamics.
        incoming = [0.0] * len(circuit.neurons)
        for edge in circuit.edges:
            incoming[edge.target] += abs(edge.weight)
        for edge in circuit.edges:
            source_sign = circuit.neurons[edge.source].sign
            total = incoming[edge.target]
            normalized = edge.weight / total if total > 1e-12 else 0.0
            self.outgoing[edge.source].append(
                (edge.target, normalized * source_sign * synapse_scale)
            )

        self.last_stimulus: RetinaStimulus | None = None
        self.last_entry_drive = [0.0] * len(circuit.neurons)
        self.previous_on_field: tuple[tuple[float, ...], ...] | None = None
        self.previous_off_field: tuple[tuple[float, ...], ...] | None = None

    def reset(self) -> None:
        self.state = [0.0] * len(self.state)
        self.last_stimulus = None
        self.last_entry_drive = [0.0] * len(self.state)
        self.previous_on_field = None
        self.previous_off_field = None

    def step(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        self.last_stimulus = stimulus
        drive = self._entry_drive(stimulus)
        self.last_entry_drive = drive

        # A real extracted pathway can be many synapses deep. One Euler update
        # per frame would attenuate a signal before it can reach T4/T5. Multiple
        # bounded micro-steps let activity propagate through the selected graph
        # while retaining leak and temporal state between visual frames.
        for _ in range(self.micro_steps):
            recurrent = [0.0] * len(self.state)
            for source, outgoing in enumerate(self.outgoing):
                activity = self.state[source]
                if activity <= 1e-12:
                    continue
                for target, weight in outgoing:
                    recurrent[target] += activity * weight

            next_state = [0.0] * len(self.state)
            for index in range(len(next_state)):
                target = max(0.0, math.tanh(drive[index] + recurrent[index]))
                next_state[index] = (1.0 - self.leak) * self.state[index] + self.leak * target
            self.state = next_state

        self.previous_on_field = stimulus.on_field
        self.previous_off_field = stimulus.off_field
        return tuple(self.state)

    def _entry_drive(self, stimulus: RetinaStimulus) -> list[float]:
        drive = [0.0] * len(self.state)
        if self.circuit.has_spatial_mapping:
            for neuron in self.circuit.l1_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.on_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_on_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, current + self.temporal_gain * (current - previous))
            for neuron in self.circuit.l2_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.off_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_off_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, current + self.temporal_gain * (current - previous))
        else:
            on_strength = max((max(row) for row in stimulus.on_field), default=0.0)
            off_strength = max((max(row) for row in stimulus.off_field), default=0.0)
            previous_on = max((max(row) for row in self.previous_on_field), default=0.0)
            previous_off = max((max(row) for row in self.previous_off_field), default=0.0)
            on_strength = max(0.0, on_strength + self.temporal_gain * (on_strength - previous_on))
            off_strength = max(0.0, off_strength + self.temporal_gain * (off_strength - previous_off))
            for neuron in self.circuit.l1_inputs:
                drive[neuron] = on_strength
            for neuron in self.circuit.l2_inputs:
                drive[neuron] = off_strength
        return drive

    def decision(self, minimum_confidence: float = 0.20) -> VisualDecision:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        t4 = self._directional_activity(self.circuit.t4_outputs)
        t5 = self._directional_activity(self.circuit.t5_outputs)
        up = max(0.0, t4[2]) + max(0.0, t5[2])
        down = max(0.0, t4[3]) + max(0.0, t5[3])
        total = up + down
        confidence = 0.0 if total <= 1e-12 else abs(up - down) / total
        coherence = self.last_stimulus.coherence if self.last_stimulus else 0.0
        return VisualDecision(up, down, confidence, confidence < minimum_confidence or coherence < 0.20)

    def entry_activity_grid(self, width: int = 32, height: int = 16) -> tuple[tuple[float, ...], ...]:
        if width < 2 or height < 2:
            raise ValueError("grid dimensions are too small")
        grid = [[0.0] * width for _ in range(height)]
        counts = [[0] * width for _ in range(height)]
        for neuron_index in self.circuit.l1_inputs + self.circuit.l2_inputs:
            neuron = self.circuit.neurons[neuron_index]
            if neuron.spatial_x is None or neuron.spatial_y is None:
                continue
            x = min(width - 1, max(0, int(round(neuron.spatial_x * (width - 1)))))
            y = min(height - 1, max(0, int(round(neuron.spatial_y * (height - 1)))))
            grid[y][x] += self.state[neuron_index]
            counts[y][x] += 1
        return tuple(tuple(value / max(1, counts[y][x]) for x, value in enumerate(row)) for y, row in enumerate(grid))

    def _directional_activity(self, groups: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        return tuple(sum(self.state[index] for index in group) / max(1, len(group)) for group in groups)


class FlyVisualPredictionAgent:
    """BNB agent whose sensory interface is an artificial visual field."""

    def __init__(self, circuit: VisualCircuit, retina_width: int = 32, retina_height: int = 16, confidence_threshold: float = 0.20) -> None:
        self.retina = BNBMarketRetina(retina_width, retina_height)
        self.visual = MaleCNSVisualSystem(circuit)
        self.confidence_threshold = confidence_threshold

    def reset(self) -> None:
        self.visual.reset()

    def perceive(self, prices: tuple[float, ...]) -> tuple[RetinaStimulus, VisualDecision]:
        stimulus = self.retina.encode(prices)
        self.visual.step(stimulus)
        return stimulus, self.visual.decision(self.confidence_threshold)

    @property
    def neuron_count(self) -> int:
        return len(self.visual.circuit.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.visual.circuit.edges)


def _sample_field(field: tuple[tuple[float, ...], ...] | None, spatial_x: float | None, spatial_y: float | None) -> float:
    if spatial_x is None or spatial_y is None or not field or not field[0]:
        return 0.0
    height = len(field)
    width = len(field[0])
    x = min(width - 1, max(0.0, spatial_x * (width - 1)))
    y = min(height - 1, max(0.0, spatial_y * (height - 1)))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(width - 1, x0 + 1), min(height - 1, y0 + 1)
    fx, fy = x - x0, y - y0
    return field[y0][x0] * (1.0 - fx) * (1.0 - fy) + field[y0][x1] * fx * (1.0 - fy) + field[y1][x0] * (1.0 - fx) * fy + field[y1][x1] * fx * fy
