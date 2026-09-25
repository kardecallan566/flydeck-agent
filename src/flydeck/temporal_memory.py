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
