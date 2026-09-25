from __future__ import annotations


class SparseEligibilityTrace:
    """Bounded sparse eligibility trace for local, reward-gated updates."""

    def __init__(self, decay: float = 0.92, maximum: float = 1.0) -> None:
        if not 0.0 <= decay < 1.0 or maximum <= 0.0:
            raise ValueError("invalid eligibility parameters")
        self.decay = decay
        self.maximum = maximum
        self._values: dict[int, float] = {}

    @property
    def values(self) -> dict[int, float]:
        return dict(self._values)

    def reset(self) -> None:
        self._values.clear()

    def step(self, active: dict[int, float]) -> None:
        for key in list(self._values):
            self._values[key] *= self.decay
            if abs(self._values[key]) < 1e-5:
                del self._values[key]
        for key, value in active.items():
            self._values[key] = max(-self.maximum, min(self.maximum, self._values.get(key, 0.0) + value))

    def reinforce(self, reward: float, learning_rate: float = 0.01) -> dict[int, float]:
        updates = {key: max(-learning_rate, min(learning_rate, learning_rate * reward * value)) for key, value in self._values.items()}
        return updates
