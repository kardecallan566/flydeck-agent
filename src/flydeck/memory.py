from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Experience:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    done: bool


class Memory:
    """Small bounded replay memory with predictable memory usage."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._items: deque[Experience] = deque(maxlen=capacity)

    def add(
        self,
        observation: tuple[float, ...],
        action: int,
        reward: float,
        next_observation: tuple[float, ...],
        done: bool,
    ) -> None:
        self._items.append(
            Experience(observation, action, reward, next_observation, done)
        )

    def last(self) -> Experience | None:
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
