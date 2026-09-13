from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Connection:
    source: int
    target: int
    weight: float


class SparseNetwork:
    """Tiny recurrent network using only explicitly stored connections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        density: float = 0.15,
        seed: int = 42,
    ) -> None:
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("network sizes must be >= 1")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = random.Random(seed)

        self._input_connections = self._connect(
            input_size, hidden_size, density, rng
        )
        self._recurrent_connections = self._connect(
            hidden_size, hidden_size, density, rng
        )
        self._output_connections = self._connect(
            hidden_size, output_size, density, rng
        )
        self._state = [0.0] * hidden_size

    @staticmethod
    def _connect(
        source_size: int, target_size: int, density: float, rng: random.Random
    ) -> list[Connection]:
        connections: list[Connection] = []
        for source in range(source_size):
            for target in range(target_size):
                if rng.random() <= density:
                    connections.append(
                        Connection(source, target, rng.uniform(-1.0, 1.0))
                    )
        return connections

    @staticmethod
    def _activate(value: float) -> float:
        return math.tanh(value)

    def step(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        if len(observation) != self.input_size:
            raise ValueError("observation size does not match network input")

        next_state = [0.0] * self.hidden_size
        for connection in self._input_connections:
            next_state[connection.target] += observation[connection.source] * connection.weight
        for connection in self._recurrent_connections:
            next_state[connection.target] += self._state[connection.source] * connection.weight

        self._state = [self._activate(value) for value in next_state]

        output = [0.0] * self.output_size
        for connection in self._output_connections:
            output[connection.target] += self._state[connection.source] * connection.weight
        return tuple(output)

    def reset(self) -> None:
        self._state = [0.0] * self.hidden_size

    @property
    def connection_count(self) -> int:
        return (
            len(self._input_connections)
            + len(self._recurrent_connections)
            + len(self._output_connections)
        )
