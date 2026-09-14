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

    def __init__(self, circuit: VisualCircuit, leak: float = 0.15, synapse_scale: float = 0.015) -> None:
        if not 0.0 < leak <= 1.0 or synapse_scale <= 0:
            raise ValueError("invalid visual dynamics")
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("visual circuit requires L1 and L2 entry neurons")
        if len(circuit.t4_outputs) != 4 or len(circuit.t5_outputs) != 4:
            raise ValueError("visual circuit requires four T4 and four T5 output groups")
        self.circuit = circuit
        self.leak = leak
        self.synapse_scale = synapse_scale
        self.state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]
        for edge in circuit.edges:
            sign = circuit.neurons[edge.source].sign
            # Missing transmitter information remains neutral. We do not guess
            # an excitatory sign just to make the circuit more active.
            self.outgoing[edge.source].append((edge.target, edge.weight * sign * synapse_scale))
        self.last_stimulus: RetinaStimulus | None = None

    def reset(self) -> None:
        self.state = [0.0] * len(self.state)
        self.last_stimulus = None

    def step(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        self.last_stimulus = stimulus
        drive = [0.0] * len(self.state)
        on_strength = max((max(row) for row in stimulus.on_field), default=0.0)
        off_strength = max((max(row) for row in stimulus.off_field), default=0.0)

        # L1 and L2 are the biological entry points of the ON/OFF motion
        # streams. The retina supplies contrast; the MaleCNS graph supplies the
        # subsequent transformation.
        for neuron in self.circuit.l1_inputs:
            drive[neuron] += on_strength
        for neuron in self.circuit.l2_inputs:
            drive[neuron] += off_strength

        recurrent = [0.0] * len(self.state)
        for source, outgoing in enumerate(self.outgoing):
            activity = self.state[source]
            if abs(activity) < 1e-12:
                continue
            for target, weight in outgoing:
                recurrent[target] += activity * weight

        next_state = [0.0] * len(self.state)
        for index in range(len(next_state)):
            target = math.tanh(drive[index] + recurrent[index])
            next_state[index] = (1.0 - self.leak) * self.state[index] + self.leak * target
        self.state = next_state
        return tuple(self.state)

    def decision(self, minimum_confidence: float = 0.20) -> VisualDecision:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        t4 = self._directional_activity(self.circuit.t4_outputs)
        t5 = self._directional_activity(self.circuit.t5_outputs)

        # T4/T5 a,b,c,d correspond to front-to-back, back-to-front, upward and
        # downward motion. The artificial BNB retina uses the vertical axis.
        up = max(0.0, t4[2]) + max(0.0, t5[2])
        down = max(0.0, t4[3]) + max(0.0, t5[3])
        total = up + down
        confidence = 0.0 if total <= 1e-12 else abs(up - down) / total
        coherence = self.last_stimulus.coherence if self.last_stimulus else 0.0
        return VisualDecision(
            up_score=up,
            down_score=down,
            confidence=confidence,
            wait=confidence < minimum_confidence or coherence < 0.20,
        )

    def _directional_activity(self, groups: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        return tuple(
            sum(self.state[index] for index in group) / max(1, len(group))
            for group in groups
        )


class FlyVisualPredictionAgent:
    """BNB agent whose sensory interface is an artificial visual field."""

    def __init__(
        self,
        circuit: VisualCircuit,
        retina_width: int = 32,
        retina_height: int = 16,
        confidence_threshold: float = 0.20,
    ) -> None:
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
