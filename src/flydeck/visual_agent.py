from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None

from .central_complex import CentralComplexState, CentralComplexSystem
from .directional_mechanism import SpatialOffsetDirectionalMechanism
from .lptc_system import LPTCOutput, LobulaPlateTangentialSystem
from .market_retina import BNBMarketRetina, RetinaStimulus
from .receptive_fields import ReceptiveField, infer_receptive_fields
from .synaptic_adaptation import SynapticAdaptation
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class VisualDecision:
    up_score: float
    down_score: float
    confidence: float
    wait: bool
    consensus: float = 1.0
    conflict: float = 0.0
    arousal: float = 0.20
    fast_bias: float = 0.0
    slow_bias: float = 0.0


class MaleCNSVisualSystem:
    """Run the compact MaleCNS motion pathway with Drosophila higher brain centers."""

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
        mutual_inhibition_gamma: float = 0.15,
        trend_memory_beta: float = 0.05,
        ablate_spatial: bool = False,
        ablate_temporal: bool = False,
        ablate_slow_inhibition: bool = False,
        ablate_mutual_inhibition: bool = False,
        ablate_t4_t5: bool = False,
        # Multi-evidence combination weights
        weight_directional: float = 0.25,
        weight_velocity: float = 0.25,
        weight_on_off_balance: float = 0.35,
        weight_trend: float = 0.15,
        # Phase 5: Drosophila Higher Brain Centers & Adaptation
        ablate_adaptation: bool = False,
        ablate_lptc: bool = False,
        ablate_working_memory: bool = False,
        ablate_neuromodulation: bool = False,
        ablate_conflict_engine: bool = False,
        conflict_threshold: float = 0.35,
    ) -> None:
        if (
            not 0.0 < leak <= 1.0
            or synapse_scale <= 0.0
            or temporal_gain < 0.0
            or micro_steps < 1
            or not 0.0 < fast_alpha <= 1.0
            or not 0.0 < slow_inhibition_alpha <= 1.0
            or receptive_field_iterations < 1
            or not 0.0 <= mutual_inhibition_gamma <= 1.0
            or not 0.0 <= trend_memory_beta <= 1.0
        ):
            raise ValueError("invalid visual dynamics")
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("visual circuit requires L1 and L2 entry neurons")
        if len(circuit.t4_outputs) != 4 or len(circuit.t5_outputs) != 4:
            raise ValueError("visual circuit requires four T4 and four T5 output groups")

        self.circuit = circuit
        self.leak = leak
        self.synapse_scale = synapse_scale
        self.temporal_gain = 0.0 if ablate_temporal else temporal_gain
        self.micro_steps = micro_steps
        self.fast_alpha = fast_alpha
        self.slow_inhibition_alpha = fast_alpha if ablate_slow_inhibition else slow_inhibition_alpha
        self.receptive_field_iterations = receptive_field_iterations
        self.mutual_inhibition_gamma = 0.0 if ablate_mutual_inhibition else mutual_inhibition_gamma
        self.trend_memory_beta = trend_memory_beta
        self.ablate_spatial = ablate_spatial
        self.ablate_temporal = ablate_temporal
        self.ablate_slow_inhibition = ablate_slow_inhibition
        self.ablate_mutual_inhibition = ablate_mutual_inhibition
        self.ablate_t4_t5 = ablate_t4_t5
        self.ablate_adaptation = ablate_adaptation
        self.ablate_lptc = ablate_lptc
        self.ablate_working_memory = ablate_working_memory
        self.ablate_neuromodulation = ablate_neuromodulation
        self.ablate_conflict_engine = ablate_conflict_engine
        self.conflict_threshold = conflict_threshold

        self.trend_bias = 0.0
        self.weight_directional = weight_directional
        self.weight_velocity = weight_velocity
        self.weight_on_off_balance = weight_on_off_balance
        self.weight_trend = weight_trend
        self.state = [0.0] * len(circuit.neurons)
        self.filtered_state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]

        # Higher brain systems
        self.synaptic_adaptation = SynapticAdaptation(
            len(circuit.neurons),
            enabled=not ablate_adaptation,
        )
        self.lptc = LobulaPlateTangentialSystem(
            enabled=not ablate_lptc,
        )
        self.central_complex = CentralComplexSystem(
            enabled=not ablate_working_memory,
        )
        self.last_lptc_out: LPTCOutput | None = None
        self.last_cx_state: CentralComplexState | None = None

        # Normalize incoming synapse mass
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

        base_rfs = receptive_fields if receptive_fields is not None else infer_receptive_fields(
            circuit,
            iterations=receptive_field_iterations,
        )
        if ablate_spatial:
            self.receptive_fields = {
                idx: ReceptiveField(
                    x=rf.x,
                    y=rf.y,
                    excitatory_x=rf.x,
                    excitatory_y=rf.y,
                    inhibitory_x=rf.x,
                    inhibitory_y=rf.y,
                    excitatory_mass=rf.excitatory_mass,
                    inhibitory_mass=rf.inhibitory_mass,
                )
                for idx, rf in base_rfs.items()
            }
        else:
            self.receptive_fields = base_rfs

        self.t4_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=self.slow_inhibition_alpha,
        )
        self.t5_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=self.slow_inhibition_alpha,
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
        self.trend_bias = 0.0
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
        self.synaptic_adaptation.reset()
        self.lptc.reset()
        self.central_complex.reset()
        self.last_lptc_out = None
        self.last_cx_state = None

    def step(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        self.last_stimulus = stimulus
        drive = self._entry_drive(stimulus)
        self.last_entry_drive = drive

        if self._has_numpy:
            drive_np = np.array(drive, dtype=np.float32)
            leak = self.leak
            for _ in range(self.micro_steps):
                self._filtered_state_np += self._alphas_np * (self._state_np - self._filtered_state_np)
                # Synaptic vesicle depression (adaptation) modulates outgoing synaptic transmission
                adapted_sources = self.synaptic_adaptation.modulate_np(self._filtered_state_np)
                contributions = adapted_sources[self._edge_sources] * self._norm_edge_weights
                recurrent = np.bincount(self._edge_targets, weights=contributions, minlength=len(self.state))
                target_act = np.maximum(0.0, np.tanh(drive_np + recurrent))
                self._state_np = (1.0 - leak) * self._state_np + leak * target_act
            self.state = self._state_np.tolist()
        else:
            for _ in range(self.micro_steps):
                recurrent = [0.0] * len(self.state)
                for source, outgoing in enumerate(self.outgoing):
                    activity = self.state[source]
                    if activity <= 1e-12:
                        self.filtered_state[source] *= self._alpha(source)
                        continue
                    alpha = self._alpha(source)
                    self.filtered_state[source] += alpha * (activity - self.filtered_state[source])
                    adapted_act = self.synaptic_adaptation.step(self.filtered_state)[source]
                    for target, weight in outgoing:
                        recurrent[target] += adapted_act * weight

                next_state = [0.0] * len(self.state)
                for index in range(len(next_state)):
                    target = max(0.0, math.tanh(drive[index] + recurrent[index]))
                    next_state[index] = (1.0 - self.leak) * self.state[index] + self.leak * target
                self.state = next_state

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

        # Lobula Plate Tangential Cells (LPTC) wide-field opponent pooling
        self.last_lptc_out = self.lptc.step(
            self.last_directional_t4,
            self.last_directional_t5,
        )

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
        gain = self.temporal_gain
        if self.circuit.has_spatial_mapping:
            for neuron in self.circuit.l1_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.on_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_on_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, current + gain * (current - previous))
            for neuron in self.circuit.l2_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.off_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_off_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, current + gain * (current - previous))
        else:
            on_strength = max((max(row) for row in stimulus.on_field), default=0.0)
            off_strength = max((max(row) for row in stimulus.off_field), default=0.0)
            previous_on = max((max(row) for row in self.previous_on_field), default=0.0)
            previous_off = max((max(row) for row in self.previous_off_field), default=0.0)
            on_strength = max(0.0, on_strength + gain * (on_strength - previous_on))
            off_strength = max(0.0, off_strength + gain * (off_strength - previous_off))
            for neuron in self.circuit.l1_inputs:
                drive[neuron] = on_strength
            for neuron in self.circuit.l2_inputs:
                drive[neuron] = off_strength
        return drive

    def decision(self, minimum_confidence: float = 0.20) -> VisualDecision:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        # === Evidence Stream 1: T4/T5 & LPTC Wide-Field Directional Flow ===
        if self.ablate_t4_t5:
            half = len(self.state) // 2
            dir_up = sum(self.state[:half]) / max(1, half)
            dir_down = sum(self.state[half:]) / max(1, len(self.state) - half)
            dir_tot = dir_up + dir_down
            dir_signal = (dir_up - dir_down) / dir_tot if dir_tot > 1e-12 else 0.0
            lptc_signal = dir_signal
        else:
            if not self.ablate_lptc and self.last_lptc_out is not None:
                lptc_signal = self.last_lptc_out.vs_net
                dir_up = self.last_lptc_out.vs_up
                dir_down = self.last_lptc_out.vs_down
                dir_signal = lptc_signal
            else:
                t4 = self._directional_activity(self.circuit.t4_outputs)
                t5 = self._directional_activity(self.circuit.t5_outputs)
                dir_up = max(0.0, t4[2]) + max(0.0, t5[2])
                dir_down = max(0.0, t4[3]) + max(0.0, t5[3])
                # Lateral inhibition
                gamma = self.mutual_inhibition_gamma
                dir_up = max(0.0, dir_up - gamma * dir_down)
                dir_down = max(0.0, dir_down - gamma * dir_up)
                dir_total = dir_up + dir_down
                dir_signal = (dir_up - dir_down) / dir_total if dir_total > 1e-12 else 0.0
                lptc_signal = dir_signal

        # === Evidence Stream 2: Retina Kinematics (Velocity & Acceleration) ===
        velocity = self.last_stimulus.velocity if self.last_stimulus else 0.0
        acceleration = self.last_stimulus.acceleration if self.last_stimulus else 0.0
        short_velocity = self.last_stimulus.short_velocity if self.last_stimulus else velocity
        # Combine multi-scale velocity with acceleration
        vel_signal = max(-1.0, min(1.0, 0.65 * velocity + 0.35 * short_velocity + 0.25 * acceleration))

        # === Evidence Stream 3: ON/OFF Circuit Balance (L1 vs L2) ===
        l1_activity = sum(self.state[n] for n in self.circuit.l1_inputs)
        l2_activity = sum(self.state[n] for n in self.circuit.l2_inputs)
        l1_count = max(1, len(self.circuit.l1_inputs))
        l2_count = max(1, len(self.circuit.l2_inputs))
        on_mean = l1_activity / l1_count
        off_mean = l2_activity / l2_count
        balance_total = on_mean + off_mean
        balance_signal = (on_mean - off_mean) / balance_total if balance_total > 1e-12 else 0.0

        # === Evidence Stream 4: Central Complex Working Memory & Neuromodulation ===
        volatility = self.last_stimulus.volatility_contrast if self.last_stimulus else 0.0035
        coherence = self.last_stimulus.coherence if self.last_stimulus else 0.5

        cx_state = self.central_complex.step(
            sensory_signal=0.5 * lptc_signal + 0.5 * vel_signal,
            volatility=volatility,
            coherence=coherence,
        )
        self.last_cx_state = cx_state

        if self.ablate_working_memory:
            beta = self.trend_memory_beta
            instant_bias = dir_signal * 0.5 + vel_signal * 0.5
            self.trend_bias = (1.0 - beta) * self.trend_bias + beta * instant_bias
            trend_signal = max(-1.0, min(1.0, self.trend_bias))
        else:
            trend_signal = cx_state.attractor_heading

        # === Multi-Evidence Weighted Combination ===
        w_d = self.weight_directional
        w_v = self.weight_velocity
        w_b = self.weight_on_off_balance
        w_t = self.weight_trend
        w_total = w_d + w_v + w_b + w_t
        if w_total > 1e-12:
            combined = (w_d * dir_signal + w_v * vel_signal + w_b * balance_signal + w_t * trend_signal) / w_total
        else:
            combined = 0.0

        # === Consensus & Conflict Engine ===
        streams = [dir_signal, vel_signal, balance_signal, trend_signal]
        conflicts = []
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                s_i, s_j = streams[i], streams[j]
                if (s_i > 0.08 and s_j < -0.08) or (s_i < -0.08 and s_j > 0.08):
                    conflicts.append(abs(s_i - s_j))

        conflict_val = sum(conflicts) / len(conflicts) if conflicts else 0.0
        consensus_val = max(0.0, 1.0 - conflict_val)

        # Convert combined signal [-1, +1] to up/down scores [0, 1]
        effective_up = max(0.0, combined)
        effective_down = max(0.0, -combined)
        confidence = abs(combined)

        # Adaptive threshold: Coherence + Neuromodulatory Arousal + Conflict
        adaptive_threshold = minimum_confidence
        if coherence < 0.25:
            adaptive_threshold += (0.25 - coherence) * 0.20

        if not self.ablate_neuromodulation:
            # High octopaminergic arousal in choppy/turbulent regimes increases caution
            adaptive_threshold += max(0.0, cx_state.arousal - 0.20) * 0.15

        if not self.ablate_conflict_engine:
            # Conflict smoothly elevates the confidence requirement rather than a hard veto
            adaptive_threshold += conflict_val * 0.20

        wait = confidence < adaptive_threshold

        return VisualDecision(
            up_score=effective_up,
            down_score=effective_down,
            confidence=confidence,
            wait=wait,
            consensus=consensus_val,
            conflict=conflict_val,
            arousal=cx_state.arousal,
            fast_bias=cx_state.fast_bias,
            slow_bias=cx_state.slow_bias,
        )

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

    def __init__(
        self,
        circuit: VisualCircuit,
        retina_width: int = 32,
        retina_height: int = 16,
        confidence_threshold: float = 0.15,
        receptive_fields: dict[int, ReceptiveField] | None = None,
        mutual_inhibition_gamma: float = 0.15,
        trend_memory_beta: float = 0.05,
        ablate_spatial: bool = False,
        ablate_temporal: bool = False,
        ablate_slow_inhibition: bool = False,
        ablate_mutual_inhibition: bool = False,
        ablate_t4_t5: bool = False,
        weight_directional: float = 0.25,
        weight_velocity: float = 0.25,
        weight_on_off_balance: float = 0.35,
        weight_trend: float = 0.15,
        ablate_adaptation: bool = False,
        ablate_lptc: bool = False,
        ablate_working_memory: bool = False,
        ablate_neuromodulation: bool = False,
        ablate_conflict_engine: bool = False,
        conflict_threshold: float = 0.35,
    ) -> None:
        self.retina = BNBMarketRetina(retina_width, retina_height)
        self.visual = MaleCNSVisualSystem(
            circuit,
            receptive_fields=receptive_fields,
            mutual_inhibition_gamma=mutual_inhibition_gamma,
            trend_memory_beta=trend_memory_beta,
            ablate_spatial=ablate_spatial,
            ablate_temporal=ablate_temporal,
            ablate_slow_inhibition=ablate_slow_inhibition,
            ablate_mutual_inhibition=ablate_mutual_inhibition,
            ablate_t4_t5=ablate_t4_t5,
            weight_directional=weight_directional,
            weight_velocity=weight_velocity,
            weight_on_off_balance=weight_on_off_balance,
            weight_trend=weight_trend,
            ablate_adaptation=ablate_adaptation,
            ablate_lptc=ablate_lptc,
            ablate_working_memory=ablate_working_memory,
            ablate_neuromodulation=ablate_neuromodulation,
            ablate_conflict_engine=ablate_conflict_engine,
            conflict_threshold=conflict_threshold,
        )
        self.confidence_threshold = confidence_threshold

    def reset(self) -> None:
        self.visual.reset()

    def perceive(
        self,
        prices: tuple[float, ...],
        volumes: tuple[float, ...] | None = None,
    ) -> tuple[RetinaStimulus, VisualDecision]:
        stimulus = self.retina.encode(prices, volumes=volumes)
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
