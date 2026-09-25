from __future__ import annotations

from dataclasses import dataclass
import math

from .bnb_prediction import Prediction


@dataclass(frozen=True, slots=True)
class Episode:
    vector: tuple[float, ...]
    action: Prediction
    reward: float
    regime: str


class BoundedEpisodicMemory:
    """Small FIFO memory of compact states; frozen during evaluation."""

    def __init__(self, capacity: int = 256, learning_enabled: bool = True) -> None:
        if capacity < 8:
            raise ValueError("episodic capacity must be at least eight")
        self.capacity = capacity
        self.learning_enabled = learning_enabled
        self._episodes: list[Episode] = []

    @property
    def size(self) -> int:
        return len(self._episodes)

    def reset(self, preserve_memory: bool = False) -> None:
        if not preserve_memory:
            self._episodes.clear()

    def set_learning(self, enabled: bool) -> None:
        self.learning_enabled = enabled

    def novelty(self, vector: tuple[float, ...]) -> float:
        if not self._episodes or not vector:
            return 1.0
        distances = [_distance(vector, episode.vector) for episode in self._episodes[-64:]]
        nearest = min(distances)
        return max(0.0, min(1.0, nearest / math.sqrt(max(1, len(vector)))) )

    def retrieve(self, vector: tuple[float, ...], regime: str, limit: int = 8) -> tuple[Episode, ...]:
        compatible = [episode for episode in self._episodes if episode.regime == regime]
        compatible.sort(key=lambda episode: _distance(vector, episode.vector))
        return tuple(compatible[:limit])

    def add(self, vector: tuple[float, ...], action: Prediction, reward: float, regime: str) -> None:
        if not self.learning_enabled or not vector or action == Prediction.WAIT:
            return
        self._episodes.append(Episode(tuple(vector), action, max(-1.0, min(1.0, reward)), regime))
        if len(self._episodes) > self.capacity:
            del self._episodes[: len(self._episodes) - self.capacity]


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 1.0
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(size)) / size)
