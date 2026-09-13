from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: tuple[float, ...]
    reward: float
    done: bool


class Environment(Protocol):
    observation_size: int
    action_size: int

    def reset(self) -> tuple[float, ...]: ...

    def step(self, action: int) -> StepResult: ...


class CounterEnvironment:
    """Minimal environment used to exercise the agent without dependencies."""

    observation_size = 1
    action_size = 2

    def __init__(self, target: int = 10, max_steps: int = 32) -> None:
        self.target = target
        self.max_steps = max_steps
        self._value = 0
        self._steps = 0

    def reset(self) -> tuple[float, ...]:
        self._value = 0
        self._steps = 0
        return (0.0,)

    def step(self, action: int) -> StepResult:
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        self._value += 1 if action == 1 else -1
        self._steps += 1
        distance = abs(self.target - self._value)
        reward = 1.0 if self._value == self.target else -0.01 * distance
        done = self._value == self.target or self._steps >= self.max_steps
        observation = (self._value / max(abs(self.target), 1),)
        return StepResult(observation, reward, done)
