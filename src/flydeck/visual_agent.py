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
    """Run the extracted L1/L2 -> T4/T5 MaleCNS pathway."""

    def __init__(self, circuit: VisualCircuit, leak: float = 0.15, synapse_scale: float = 0.015) -> None:
        if not 0.0 < leak <= 1.0 or synapse_scale <= 0:
            raise ValueError("invalid visual dynamics")
        if len(circuit.on_inputs) != 4 or len(circuit.off_inputs) != 4:
            raise ValueError("visual circuit requires four ON and four OFF input pools")
        if len(circuit.t4_outputs) != 4 or len(circuit.t5_outputs) != 4:
            raise ValueError("visual circuit requires four T4 and four T5 output pools")
        self.circuit = circuit
        self.leak = leak
        self.synapse_scale = synapse_scale
        self.state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]
        for edge in circuit.edges:
            source_sign = circuit.neurons[edge.source].sign
            self.outgoing[edge.source].append((edge.target, edge.weight * source_sign * synapse_scale))
        self.last_stimulus: RetinaStimulus | None = None

    def reset(self) -> None:
        self.state = [0.0] * len(self.state)
        self.last_stimulus = None

    def step(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        self.last_stimulus = stimulus
        drive = [0.0] * len(self.state)
        for channel, value in enumerate(stimulus.directions):
            for neuron in self.circuit.on_inputs[channel]:
                if self.circuit.neurons[neuron].cell_type.lower() == "l1":
                    drive[neuron] += value
            for neuron in self.circuit.off_inputs[channel]:
                if self.circuit.neurons[neuron].cell_type.lower() == "l2":
                    drive[neuron] += value

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

    def decision(self) -> VisualDecision:
        t4 = self._directional_activity(self.circuit.t4_outputs)
        t5 = self._directional_activity(self.circuit.t5_outputs)
        # T4/T5 a,b,c,d encode front-to-back, back-to-front, upward, downward.
        # Only c/d are mapped to the vertical price axis of the artificial retina.
        up = max(0.0, t4[2]) + max(0.0, t5[2])
        down = max(0.0, t4[3]) + max(0.0, t5[3])
        total = up + down
        confidence = 0.0 if total <= 1e-12 else abs(up - down) / total
        return VisualDecision(up, down, confidence, confidence < 0.20)

    def _directional_activity(self, groups: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        return tuple(sum(self.state[index] for index in group) / max(1, len(group)) for group in groups)


class FlyVisualPredictionAgent:
    """BNB agent whose sensory input is an artificial visual field."""

    def __init__(self, circuit: VisualCircuit, retina_width: int = 32, retina_height: int = 16) -> None:
        self.retina = BNBMarketRetina(retina_width, retina_height)
        self.visual = MaleCNSVisualSystem(circuit)

    def reset(self) -> None:
        self.visual.reset()

    def perceive(self, prices: tuple[float, ...]) -> tuple[RetinaStimulus, VisualDecision]:
        stimulus = self.retina.encode(prices)
        self.visual.step(stimulus)
        return stimulus, self.visual.decision()

    @property
    def neuron_count(self) -> int:
        return len(self.visual.circuit.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.visual.circuit.edges)
