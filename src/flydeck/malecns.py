from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

MALECNS_BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
)
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"

NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}


@dataclass(frozen=True, slots=True)
class MaleCNSNeuron:
    body_id: int
    role: str = "internal"
    neurotransmitter: str | None = None


@dataclass(frozen=True, slots=True)
class MaleCNSEdge:
    source: int
    target: int
    weight: float


@dataclass(frozen=True, slots=True)
class MaleCNSCircuit:
    """Compact, indexed MaleCNS subgraph suitable for online simulation."""

    neurons: tuple[MaleCNSNeuron, ...]
    edges: tuple[MaleCNSEdge, ...]
    input_neurons: tuple[tuple[int, ...], ...]
    output_neurons: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, path: str | Path) -> "MaleCNSCircuit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        neurons = tuple(
            MaleCNSNeuron(
                body_id=int(item["body_id"]),
                role=item.get("role", "internal"),
                neurotransmitter=item.get("neurotransmitter"),
            )
            for item in payload["neurons"]
        )
        edges = tuple(
            MaleCNSEdge(int(item[0]), int(item[1]), float(item[2]))
            for item in payload["edges"]
        )
        return cls(
            neurons=neurons,
            edges=edges,
            input_neurons=tuple(tuple(group) for group in payload["input_neurons"]),
            output_neurons=tuple(tuple(group) for group in payload["output_neurons"]),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "source": "MaleCNS v1.0",
            "neurons": [
                {"body_id": n.body_id, "role": n.role, "neurotransmitter": n.neurotransmitter}
                for n in self.neurons
            ],
            "edges": [[e.source, e.target, e.weight] for e in self.edges],
            "input_neurons": [list(group) for group in self.input_neurons],
            "output_neurons": [list(group) for group in self.output_neurons],
        }
        Path(path).write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )


def download_malecns_data(directory: str | Path) -> dict[str, Path]:
    """Download official MaleCNS inputs; raw data is never committed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "annotations": ANNOTATIONS_FILE,
        "weights": WEIGHTS_FILE,
        "neurotransmitters": NEUROTRANSMITTERS_FILE,
    }
    result: dict[str, Path] = {}
    for key, filename in files.items():
        path = directory / filename
        if not path.exists():
            urlretrieve(MALECNS_BASE_URL + filename, path)
        result[key] = path
    return result


class MaleCNSReservoir:
    """Fixed recurrent dynamics using an extracted MaleCNS circuit.

    The connectome is frozen in V1. Market features are injected into explicit
    input neuron pools and three action pools are decoded from neuron activity.
    Only the small action readout is trainable; this isolates biological
    topology before introducing synaptic plasticity.
    """

    def __init__(
        self,
        circuit: MaleCNSCircuit,
        feature_count: int,
        action_count: int = 3,
        seed: int = 42,
        leak: float = 0.20,
        activation_scale: float = 0.08,
    ) -> None:
        if feature_count < 1 or action_count < 1:
            raise ValueError("feature_count and action_count must be >= 1")
        if len(circuit.input_neurons) != feature_count:
            raise ValueError("circuit input pool count must match feature_count")
        if len(circuit.output_neurons) != action_count:
            raise ValueError("circuit output pool count must match action_count")
        if not 0.0 < leak <= 1.0 or activation_scale <= 0.0:
            raise ValueError("invalid reservoir dynamics")

        self.circuit = circuit
        self.feature_count = feature_count
        self.action_count = action_count
        self.leak = leak
        self.activation_scale = activation_scale
        self._state = [0.0] * len(circuit.neurons)
        rng = random.Random(seed)
        self._readout = [rng.uniform(-0.1, 0.1) for _ in range(action_count)]
        self._readout_bias = [0.0] * action_count
        self._outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]
        for edge in circuit.edges:
            if not 0 <= edge.source < len(self._state) or not 0 <= edge.target < len(self._state):
                raise ValueError("edge index outside circuit neuron range")
            self._outgoing[edge.source].append((edge.target, edge.weight))

    def reset(self) -> None:
        self._state = [0.0] * len(self._state)

    @staticmethod
    def _activate(value: float) -> float:
        return math.tanh(value)

    def step(self, features: tuple[float, ...]) -> tuple[float, ...]:
        if len(features) != self.feature_count:
            raise ValueError("feature count does not match MaleCNS input pools")

        drive = [0.0] * len(self._state)
        for feature, pool in zip(features, self.circuit.input_neurons):
            value = max(-1.0, min(1.0, feature)) * self.activation_scale
            for neuron in pool:
                drive[neuron] += value

        next_state = [0.0] * len(self._state)
        for source, outgoing in enumerate(self._outgoing):
            source_activity = self._state[source]
            if source_activity == 0.0:
                continue
            for target, weight in outgoing:
                next_state[target] += source_activity * weight

        for index in range(len(next_state)):
            target = self.leak * self._activate(drive[index] + next_state[index])
            next_state[index] = (1.0 - self.leak) * self._state[index] + target
        self._state = next_state

        return tuple(self.action_scores())

    def action_scores(self) -> tuple[float, ...]:
        scores = []
        for action, pool in enumerate(self.circuit.output_neurons):
            activity = sum(self._state[index] for index in pool) / max(1, len(pool))
            scores.append(activity * self._readout[action] + self._readout_bias[action])
        return tuple(scores)

    def action_activity(self, action: int) -> float:
        if not 0 <= action < self.action_count:
            raise ValueError("action is outside the output range")
        pool = self.circuit.output_neurons[action]
        return sum(self._state[index] for index in pool) / max(1, len(pool))

    def update_readout(
        self,
        action: int,
        td_error: float,
        learning_rate: float = 0.005,
        activity: float | None = None,
    ) -> None:
        if not 0 <= action < self.action_count:
            raise ValueError("action is outside the output range")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if activity is None:
            activity = self.action_activity(action)
        self._readout[action] = max(
            -2.0,
            min(2.0, self._readout[action] + learning_rate * td_error * activity),
        )
        self._readout_bias[action] = max(
            -2.0,
            min(2.0, self._readout_bias[action] + learning_rate * td_error),
        )

    @property
    def neuron_count(self) -> int:
        return len(self._state)

    @property
    def edge_count(self) -> int:
        return len(self.circuit.edges)
