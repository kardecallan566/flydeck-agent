from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetinaStimulus:
    """A 2-D artificial visual field presented to the fly."""

    on_field: tuple[tuple[float, ...], ...]
    off_field: tuple[tuple[float, ...], ...]
    directions: tuple[float, float, float, float]
    coherence: float
    velocity: float
    acceleration: float


class BNBMarketRetina:
    """Convert only the observed BNB price history into visual motion."""

    def __init__(self, width: int = 32, height: int = 16) -> None:
        if width < 4 or height < 4:
            raise ValueError("retina dimensions are too small")
        self.width = width
        self.height = height

    def encode(self, prices: tuple[float, ...]) -> RetinaStimulus:
        if len(prices) < 4 or any(price <= 0 for price in prices):
            raise ValueError("prices must contain at least four positive values")

        window = prices[-self.width :]
        lo, hi = min(window), max(window)
        span = max(hi - lo, hi * 1e-9)
        positions = [
            max(0, min(self.height - 1, int(round((price - lo) / span * (self.height - 1)))) )
            for price in window
        ]
        start = self.width - len(window)
        on = [[0.0] * self.width for _ in range(self.height)]
        off = [[0.0] * self.width for _ in range(self.height)]
        velocities: list[float] = []
        up = down = 0.0

        for i, row in enumerate(positions):
            x = start + i
            on[row][x] = 1.0
            if i == 0:
                continue
            delta = window[i] / window[i - 1] - 1.0
            velocities.append(delta)
            strength = min(1.0, abs(delta) * 100.0)
            previous = positions[i - 1]
            if delta > 0:
                up += strength
                on[previous][x] = max(on[previous][x], strength)
            elif delta < 0:
                down += strength
                off[previous][x] = max(off[previous][x], strength)

        persistence = (
            sum(a * b > 0 for a, b in zip(velocities, velocities[1:]))
            / max(1, len(velocities) - 1)
            if len(velocities) > 1
            else 0.0
        )
        reversal = 1.0 - persistence
        # Cardinal channels: right, left, up, down. Only the vertical channels
        # are mapped to price direction; horizontal channels encode persistence
        # versus reversal without inventing financial semantics.
        raw = (up * persistence, down * reversal, up, down)
        total = sum(raw)
        directions = tuple(value / total for value in raw) if total else (0.0,) * 4

        mean_velocity = sum(velocities) / max(1, len(velocities))
        acceleration = velocities[-1] - velocities[-2] if len(velocities) >= 2 else 0.0
        mean_abs = sum(abs(value) for value in velocities) / max(1, len(velocities))
        coherence = abs(mean_velocity) / max(mean_abs, 1e-12)

        return RetinaStimulus(
            on_field=tuple(tuple(min(1.0, value) for value in row) for row in on),
            off_field=tuple(tuple(min(1.0, value) for value in row) for row in off),
            directions=directions,
            coherence=max(0.0, min(1.0, coherence)),
            velocity=max(-1.0, min(1.0, mean_velocity * 100.0)),
            acceleration=max(-1.0, min(1.0, acceleration * 100.0)),
        )

    @staticmethod
    def ascii(stimulus: RetinaStimulus) -> str:
        lines: list[str] = []
        for y in range(len(stimulus.on_field) - 1, -1, -1):
            row: list[str] = []
            for x in range(len(stimulus.on_field[y])):
                on = stimulus.on_field[y][x]
                off = stimulus.off_field[y][x]
                row.append("#" if on > 0 and off == 0 else "o" if off > 0 and on == 0 else "*")
            lines.append("".join(row))
        return "\n".join(lines)

    @staticmethod
    def flatten(stimulus: RetinaStimulus) -> tuple[float, ...]:
        return (
            tuple(value for row in stimulus.on_field for value in row)
            + tuple(value for row in stimulus.off_field for value in row)
            + stimulus.directions
            + (stimulus.coherence, stimulus.velocity, stimulus.acceleration)
        )
