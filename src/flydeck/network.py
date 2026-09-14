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

    def __init__(self, input_size: int, hidden_size: int, output_size: int, density: float = 0.15, seed: int = 42) -> None:
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("network sizes must be >= 1")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = random.Random(seed)
        self._input_connections = self._connect(input_size, hidden_size, density, rng)
        self._recurrent_connections = self._connect(hidden_size, hidden_size, density, rng)
        self._output_connections = self._connect(hidden_size, output_size, density, rng)
        self._state = [0.0] * hidden_size
        self._decision_state = [0.0] * hidden_size
        self._previous_decision_state = [0.0] * hidden_size
        self._eligibility = [0.0] * len(self._output_connections)

    @staticmethod
    def _connect(source_size: int, target_size: int, density: float, rng: random.Random) -> list[Connection]:
        connections: list[Connection] = []
        for source in range(source_size):
            for target in range(target_size):
                if rng.random() <= density:
                    connections.append(Connection(source, target, rng.uniform(-1.0, 1.0)))
        connected_targets = {connection.target for connection in connections}
        for target in range(target_size):
            if target not in connected_targets:
                source = rng.randrange(source_size)
                connections.append(Connection(source, target, rng.uniform(-1.0, 1.0)))
        return connections

    @staticmethod
    def _activate(value: float) -> float:
        return math.tanh(value)

    def step(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        if len(observation) != self.input_size:
            raise ValueError("observation size does not match network input")
        self._previous_decision_state = list(self._decision_state)
        next_state = [0.0] * self.hidden_size
        for connection in self._input_connections:
            next_state[connection.target] += observation[connection.source] * connection.weight
        for connection in self._recurrent_connections:
            next_state[connection.target] += self._state[connection.source] * connection.weight
        self._state = [self._activate(value) for value in next_state]
        self._decision_state = list(self._state)
        output = [0.0] * self.output_size
        for connection in self._output_connections:
            output[connection.target] += self._state[connection.source] * connection.weight
        return tuple(output)

    def learn(self, action: int, reward: float, learning_rate: float = 0.02) -> None:
        """Legacy immediate reward update kept for non-finance environments."""
        if not 0 <= action < self.output_size:
            raise ValueError("action is outside the output range")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        competitor_scale = 1.0 / max(self.output_size - 1, 1)
        updated: list[Connection] = []
        for connection in self._output_connections:
            activation = self._decision_state[connection.source]
            direction = 1.0 if connection.target == action else -competitor_scale
            weight = connection.weight + learning_rate * reward * activation * direction
            weight = max(-2.0, min(2.0, weight))
            updated.append(Connection(connection.source, connection.target, weight))
        self._output_connections = updated

    def _learn_td(
        self,
        action: int,
        reward: float,
        next_scores: tuple[float, ...],
        done: bool,
        learning_rate: float,
        discount: float,
        trace_decay: float,
        competitive: bool,
    ) -> float:
        if not 0 <= action < self.output_size:
            raise ValueError("action is outside the output range")
        if len(next_scores) != self.output_size:
            raise ValueError("next_scores size does not match network output")
        if learning_rate <= 0 or not 0.0 < discount <= 1.0:
            raise ValueError("learning_rate must be > 0 and discount must be in (0, 1]")
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay must be between 0 and 1")

        decision_state = self._decision_state if done else self._previous_decision_state
        bootstrap = 0.0 if done else max(next_scores)
        current = sum(
            decision_state[connection.source] * connection.weight
            for connection in self._output_connections
            if connection.target == action
        )
        td_error = reward + discount * bootstrap - current
        td_error = max(-1.0, min(1.0, td_error))

        updated: list[Connection] = []
        new_eligibility: list[float] = []
        competitor_scale = 1.0 / max(self.output_size - 1, 1)
        decay = discount * trace_decay
        for index, connection in enumerate(self._output_connections):
            activation = decision_state[connection.source]
            if competitive:
                direction = 1.0 if connection.target == action else -competitor_scale
            else:
                direction = 1.0 if connection.target == action else 0.0
            trace = decay * self._eligibility[index] + activation * direction
            weight = connection.weight + learning_rate * td_error * trace
            weight = max(-2.0, min(2.0, weight))
            updated.append(Connection(connection.source, connection.target, weight))
            new_eligibility.append(trace)
        self._output_connections = updated
        self._eligibility = new_eligibility
        return td_error

    def learn_td(self, action: int, reward: float, next_scores: tuple[float, ...], done: bool, learning_rate: float = 0.005, discount: float = 0.97, trace_decay: float = 0.85) -> float:
        """Learn TD(lambda) with the legacy competitive output update."""
        return self._learn_td(action, reward, next_scores, done, learning_rate, discount, trace_decay, competitive=True)

    def learn_td_selected_action(self, action: int, reward: float, next_scores: tuple[float, ...], done: bool, learning_rate: float = 0.005, discount: float = 0.97, trace_decay: float = 0.85) -> float:
        """Learn TD(lambda) while updating only the selected action value."""
        return self._learn_td(action, reward, next_scores, done, learning_rate, discount, trace_decay, competitive=False)

    def learn_td_all_actions(self, rewards: tuple[float, ...], next_scores_by_action: tuple[tuple[float, ...], ...], done_by_action: tuple[bool, ...], learning_rate: float = 0.005, discount: float = 0.97, trace_decay: float = 0.85) -> tuple[float, ...]:
        """Learn one independent TD target for every action."""
        if len(rewards) != self.output_size:
            raise ValueError("rewards size does not match network output")
        if len(next_scores_by_action) != self.output_size:
            raise ValueError("next_scores_by_action size does not match network output")
        if len(done_by_action) != self.output_size:
            raise ValueError("done_by_action size does not match network output")
        if learning_rate <= 0 or not 0.0 < discount <= 1.0:
            raise ValueError("learning_rate must be > 0 and discount must be in (0, 1]")
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay must be between 0 and 1")
        if any(len(scores) != self.output_size for scores in next_scores_by_action):
            raise ValueError("each next score vector must match network output")

        decay = discount * trace_decay
        decision_state = self._decision_state
        td_errors: list[float] = []
        for action in range(self.output_size):
            current = sum(
                decision_state[connection.source] * connection.weight
                for connection in self._output_connections
                if connection.target == action
            )
            bootstrap = 0.0 if done_by_action[action] else max(next_scores_by_action[action])
            td_error = rewards[action] + discount * bootstrap - current
            td_errors.append(max(-1.0, min(1.0, td_error)))

        updated: list[Connection] = []
        new_eligibility: list[float] = []
        for index, connection in enumerate(self._output_connections):
            activation = decision_state[connection.source]
            action = connection.target
            trace = decay * self._eligibility[index] + activation
            weight = connection.weight + learning_rate * td_errors[action] * trace
            weight = max(-2.0, min(2.0, weight))
            updated.append(Connection(connection.source, connection.target, weight))
            new_eligibility.append(trace)
        self._output_connections = updated
        self._eligibility = new_eligibility
        return tuple(td_errors)

    def reset(self) -> None:
        self._state = [0.0] * self.hidden_size
        self._decision_state = [0.0] * self.hidden_size
        self._previous_decision_state = [0.0] * self.hidden_size
        self._eligibility = [0.0] * len(self._output_connections)

    @property
    def connection_count(self) -> int:
        return len(self._input_connections) + len(self._recurrent_connections) + len(self._output_connections)
