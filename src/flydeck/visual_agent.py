from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None

from .directional_mechanism import SpatialOffsetDirectionalMechanism
from .market_retina import BNBMarketRetina, RetinaStimulus
from .receptive_fields import infer_receptive_fields
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class VisualDecision:
    up_score: float
    down_score: float
    confidence: float
    wait: bool


class MaleCNSVisualSystem:
    """Run the compact MaleCNS motion pathway with temporal synaptic filtering."""

    def __init__(
        self,
        circuit: VisualCircuit,
        leak: float = 0.30,
        synapse_scale: float = 1.20,
        temporal_gain: float = 0.75,
        micro_steps: int = 4,
        fast_alpha: float = 0.75,
        slow_inhibition_alpha: float = 0.20,
        receptive_field_iterations: int = 12,
        receptive_fields: dict[int, ReceptiveField] | None = None,
    ) -> None:
        if (
            not 0.0 < leak <= 1.0
            or synapse_scale <= 0.0
            or temporal_gain < 0.0
            or micro_steps < 1
            or not 0.0 < fast_alpha <= 1.0
            or not 0.0 < slow_inhibition_alpha <= 1.0
            or receptive_field_iterations < 1
        ):
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
        self.fast_alpha = fast_alpha
        self.slow_inhibition_alpha = slow_inhibition_alpha
        self.receptive_field_iterations = receptive_field_iterations
        self.state = [0.0] * len(circuit.neurons)
        self.filtered_state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]

        # Normalize incoming synapse mass. The sign remains attached to the
        # source neuron, so inhibitory MaleCNS transmitters suppress downstream
        # activity without creating negative firing rates.
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

        if np is not None:
            self._has_numpy = True
            self._state_np = np.zeros(len(circuit.neurons), dtype=np.float32)
            self._filtered_state_np = np.zeros(len(circuit.neurons), dtype=np.float32)
            self._alphas_np = np.array([self._alpha(i) for i in range(len(circuit.neurons))], dtype=np.float32)
            self._edge_sources = np.array([e.source for e in circuit.edges], dtype=np.int32)
            self._edge_targets = np.array([e.target for e in circuit.edges], dtype=np.int32)
            source_signs = np.array([circuit.neurons[e.source].sign for e in circuit.edges], dtype=np.float32)
            weights = np.array([e.weight for e in circuit.edges], dtype=np.float32)
            inc = np.array(incoming, dtype=np.float32)[self._edge_targets]
            inc = np.maximum(1e-12, inc)
            self._norm_edge_weights = (weights / inc) * source_signs * synapse_scale
        else:
            self._has_numpy = False

        # Directional selectivity is computed from the connectome-derived
        # spatially offset excitation/inhibition components of each T4/T5 RF.
        self.receptive_fields = receptive_fields if receptive_fields is not None else infer_receptive_fields(
            circuit,
            iterations=receptive_field_iterations,
        )
        self.t4_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=slow_inhibition_alpha,
        )
        self.t5_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=slow_inhibition_alpha,
        )
        self.last_directional_t4 = (0.0, 0.0, 0.0, 0.0)
        self.last_directional_t5 = (0.0, 0.0, 0.0, 0.0)

        self.last_stimulus: RetinaStimulus | None = None
        self.last_entry_drive = [0.0] * len(circuit.neurons)
        self.previous_on_field: tuple[tuple[float, ...], ...] | None = None
        self.previous_off_field: tuple[tuple[float, ...], ...] | None = None

    def reset(self) -> None:
        self.state = [0.0] * len(self.state)
        self.filtered_state = [0.0] * len(self.state)
        if self._has_numpy:
            self._state_np.fill(0.0)
            self._filtered_state_np.fill(0.0)
        self.last_stimulus = None
        self.last_entry_drive = [0.0] * len(self.state)
        self.previous_on_field = None
        self.previous_off_field = None
        self.t4_directional.reset()
        self.t5_directional.reset()
        self.last_directional_t4 = (0.0, 0.0, 0.0, 0.0)
        self.last_directional_t5 = (0.0, 0.0, 0.0, 0.0)

    def step(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        self.last_stimulus = stimulus
        drive = self._entry_drive(stimulus)
        self.last_entry_drive = drive

        if self._has_numpy:
            drive_np = np.array(drive, dtype=np.float32)
            leak = self.leak
            for _ in range(self.micro_steps):
                self._filtered_state_np += self._alphas_np * (self._state_np - self._filtered_state_np)
                contributions = self._filtered_state_np[self._edge_sources] * self._norm_edge_weights
                recurrent = np.bincount(self._edge_targets, weights=contributions, minlength=len(self.state))
                target_act = np.maximum(0.0, np.tanh(drive_np + recurrent))
                self._state_np = (1.0 - leak) * self._state_np + leak * target_act
            self.state = self._state_np.tolist()
        else:
            for _ in range(self.micro_steps):
                recurrent = [0.0] * len(self.state)

                # Upstream visual dynamics use fast excitatory and slow inhibitory
                # source filtering. Directional T4/T5 computation is applied after
                # this recurrent propagation using their spatially offset RFs.
                for source, outgoing in enumerate(self.outgoing):
                    activity = self.state[source]
                    if activity <= 1e-12:
                        self.filtered_state[source] *= self._alpha(source)
                        continue
                    alpha = self._alpha(source)
                    self.filtered_state[source] += alpha * (activity - self.filtered_state[source])
                    for target, weight in outgoing:
                        recurrent[target] += self.filtered_state[source] * weight

                next_state = [0.0] * len(self.state)
                for index in range(len(next_state)):
                    target = max(0.0, math.tanh(drive[index] + recurrent[index]))
                    next_state[index] = (1.0 - self.leak) * self.state[index] + self.leak * target
                self.state = next_state

        # T4 is the ON motion detector and T5 is the OFF motion detector. Each
        # population owns its temporal trace so ON and OFF do not contaminate
        # each other's inhibition state.
        self.last_directional_t4 = self.t4_directional.step(
            stimulus,
            self.circuit.t4_outputs,
            polarity="on",
        )
        self.last_directional_t5 = self.t5_directional.step(
            stimulus,
            self.circuit.t5_outputs,
            polarity="off",
        )
        for groups, values in (
            (self.circuit.t4_outputs, self.last_directional_t4),
            (self.circuit.t5_outputs, self.last_directional_t5),
        ):
            for group, value in zip(groups, values):
                for index in group:
                    self.state[index] = value
                    if self._has_numpy:
                        self._state_np[index] = value

        self.previous_on_field = stimulus.on_field
        self.previous_off_field = stimulus.off_field
        return tuple(self.state)

    def _alpha(self, index: int) -> float:
        neuron = self.circuit.neurons[index]
        if neuron.sign < 0.0:
            return self.slow_inhibition_alpha
        return self.fast_alpha

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
