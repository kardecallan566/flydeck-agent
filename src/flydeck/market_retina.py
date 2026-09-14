from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetinaStimulus:
    """Artificial visual scene presented to the MaleCNS motion pathway."""
    on_field: tuple[tuple[float, ...], ...]
    off_field: tuple[tuple[float, ...], ...]
    directions: tuple[float, float, float, float]
    coherence: float
    velocity: float
    acceleration: float


class BNBMarketRetina:
    """Turn a causal BNB price trace into a small visual field.

    The trace is drawn as a horizontal bright/dark edge whose vertical position
    moves between consecutive observations. A rising price therefore produces
    an upward-moving ON edge; a falling price produces an upward-moving OFF edge
    in the corresponding visual channel. The field is only a representation of
    the current causal window, never future information.
    """

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
        rows = [int(round((price - lo) / span * (self.height - 1))) for price in window]
        rows = [max(0, min(self.height - 1, row)) for row in rows]

        on = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        off = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        velocities: list[float] = []
        up = down = 0.0

        start = self.width - len(window)
        for i, row in enumerate(rows):
            x = start + i
            on[row][x] = 1.0
            if i == 0:
                continue
            delta = window[i] / window[i - 1] - 1.0
            velocities.append(delta)
            strength = min(1.0, abs(delta) * 100.0)
            previous_row = rows[i - 1]
            if delta > 0:
                up += strength
                on[previous_row][x] = 1.0
            elif delta < 0:
                down += strength
                off[previous_row][x] = 1.0

        total = up + down
        vertical = up / total if total else 0.0
        horizontal = down / total if total else 0.0
        persistence = 0.0
        if len(velocities) > 1:
            persistence = sum(a * b > 0 for a, b in zip(velocities, velocities[1:])) / (len(velocities) - 1)
        directions = (0.0, horizontal * persistence, vertical, horizontal * (1.0 - persistence))
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
        """Render the actual stimulus seen by the agent for debugging."""
        lines: list[str] = []
        for y in range(len(stimulus.on_field) - 1, -1, -1):
            row = []
            for x in range(len(stimulus.on_field[y])):
                on = stimulus.on_field[y][x]
                off = stimulus.off_field[y][x]
                row.append("#" if on > 0 and off == 0 else "o" if off > 0 and on == 0 else "*")
            lines.append("".join(row))
        return "\n".join(lines)
