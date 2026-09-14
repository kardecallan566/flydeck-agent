from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RetinaStimulus:
    """Compact visual stimulus presented to the fly's visual circuit.

    The market is represented as a moving luminance edge rather than as raw
    financial indicators. ON/OFF channels encode increases/decreases and four
    directional channels encode the local motion of the price trace.
    """

    on: tuple[float, ...]
    off: tuple[float, ...]
    directions: tuple[float, float, float, float]
    coherence: float
    velocity: float
    acceleration: float


class BNBMarketRetina:
    """Turn a causal BNB price window into an artificial visual stimulus."""

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

        on = [0.0] * self.height
        off = [0.0] * self.height
        velocities: list[float] = []
        direction_votes = [0.0, 0.0, 0.0, 0.0]

        for previous, current in zip(window, window[1:]):
            delta = current / previous - 1.0
            velocities.append(delta)
            strength = min(1.0, abs(delta) * 100.0)
            if delta > 0:
                on[positions[-1]] += strength
                direction_votes[0] += strength
            elif delta < 0:
                off[positions[-1]] += strength
                direction_votes[2] += strength

        # The visual circuit has four cardinal motion channels. Price has a
        # natural vertical axis; the horizontal channels encode persistence and
        # reversal of the moving edge rather than pretending the market is a
        # literal two-dimensional visual scene.
        if velocities:
            up = sum(max(0.0, value) for value in velocities)
            down = sum(max(0.0, -value) for value in velocities)
            persistence = sum(
                1.0 for a, b in zip(velocities, velocities[1:]) if a * b > 0
            ) / max(1, len(velocities) - 1)
            reversal = 1.0 - persistence
            direction_votes[1] = up * persistence
            direction_votes[3] = down * reversal

        total = sum(direction_votes)
        if total > 0:
            directions = tuple(value / total for value in direction_votes)
        else:
            directions = (0.0, 0.0, 0.0, 0.0)

        mean_velocity = sum(velocities) / max(1, len(velocities))
        acceleration = 0.0
        if len(velocities) >= 2:
            acceleration = velocities[-1] - velocities[-2]

        mean_abs = sum(abs(value) for value in velocities) / max(1, len(velocities))
        coherence = abs(mean_velocity) / max(mean_abs, 1e-12)

        return RetinaStimulus(
            on=tuple(min(1.0, value) for value in on),
            off=tuple(min(1.0, value) for value in off),
            directions=tuple(directions),
            coherence=max(0.0, min(1.0, coherence)),
            velocity=max(-1.0, min(1.0, mean_velocity * 100.0)),
            acceleration=max(-1.0, min(1.0, acceleration * 100.0)),
        )

    def flatten(self, stimulus: RetinaStimulus) -> tuple[float, ...]:
        """Return a deterministic sensory vector for visual-circuit injection."""
        return (
            *stimulus.on,
            *stimulus.off,
            *stimulus.directions,
            stimulus.coherence,
            stimulus.velocity,
            stimulus.acceleration,
        )
