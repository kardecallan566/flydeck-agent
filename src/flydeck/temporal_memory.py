from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalMemoryState:
    fast: float = 0.0
    slow: float = 0.0
    novelty: float = 0.0


class DualTimescaleMemory:
    """Two scalar EMAs for fast motion and slow context."""

    def __init__(self, fast_alpha: float = 0.30, slow_alpha: float = 0.03) -> None:
        if not 0.0 < slow_alpha <= fast_alpha <= 1.0:
            raise ValueError("memory alphas must satisfy 0 < slow <= fast <= 1")
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self._state = TemporalMemoryState()

    @property
    def state(self) -> TemporalMemoryState:
        return self._state

    def reset(self) -> None:
        self._state = TemporalMemoryState()

    def update(self, signal: float, novelty: float = 0.0) -> TemporalMemoryState:
        fast = (1.0 - self.fast_alpha) * self._state.fast + self.fast_alpha * signal
        slow = (1.0 - self.slow_alpha) * self._state.slow + self.slow_alpha * signal
        self._state = TemporalMemoryState(fast=fast, slow=slow, novelty=max(0.0, min(1.0, novelty)))
        return self._state


from collections import deque
import math
from .bnb_prediction import Prediction
from .temporal_events import TemporalEvent


@dataclass(frozen=True, slots=True)
class TemporalMemoryContext:
    up: float = 0.0
    down: float = 0.0
    flat: float = 0.0
    attention: float = 0.0
    matches: int = 0


@dataclass(frozen=True, slots=True)
class _MemoryEntry:
    event: TemporalEvent
    label: Prediction
    reward: float
    reliability: float


class SparseTemporalMemory:
    """Bounded associative memory that stores only salient causal events."""

    def __init__(self, capacity: int = 512, top_k: int = 8, decay: float = 0.985,
                 novelty_threshold: float = 0.10) -> None:
        if capacity < 8 or top_k < 1 or not 0.0 < decay <= 1.0:
            raise ValueError("invalid temporal memory parameters")
        self.capacity = capacity
        self.top_k = top_k
        self.decay = decay
        self.novelty_threshold = novelty_threshold
        self._entries: deque[_MemoryEntry] = deque(maxlen=capacity)

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(self, event: TemporalEvent, label: Prediction, reward: float) -> None:
        bounded_reward = max(-1.0, min(1.0, reward))
        self._entries.append(_MemoryEntry(event, label, bounded_reward, bounded_reward))

    def attend(self, query: TemporalEvent) -> TemporalMemoryContext:
        if not self._entries:
            return TemporalMemoryContext()
        ranked: list[tuple[float, _MemoryEntry]] = []
        for entry in self._entries:
            similarity = self._similarity(query.state, entry.event.state)
            age = max(0, query.timestamp - entry.event.timestamp)
            age_weight = self.decay ** min(age, 10_000)
            regime_weight = 1.15 if entry.event.regime == query.regime else 0.75
            score = max(0.0, similarity) * max(0.05, entry.event.intensity) * age_weight * regime_weight
            if score >= self.novelty_threshold:
                ranked.append((score, entry))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[:self.top_k]
        if not selected:
            return TemporalMemoryContext()
        total = sum(score for score, _ in selected)
        up = down = flat = 0.0
        for score, entry in selected:
            weight = score / max(1e-12, total)
            signed = entry.reliability * weight
            if entry.label == Prediction.UP:
                up += signed
            elif entry.label == Prediction.DOWN:
                down += signed
            else:
                flat += abs(signed)
        return TemporalMemoryContext(up=up, down=down, flat=flat,
                                     attention=min(1.0, total / len(selected)), matches=len(selected))

    @staticmethod
    def _similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) * 0.5))
