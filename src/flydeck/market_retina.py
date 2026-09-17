from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetinaStimulus:
    """A 2-D artificial visual field presented to the fly with multi-layer sensory features."""

    on_field: tuple[tuple[float, ...], ...]
    off_field: tuple[tuple[float, ...], ...]
    directions: tuple[float, float, float, float]
    coherence: float
    velocity: float
    acceleration: float
    volume_contrast: float = 1.0
    volatility_contrast: float = 1.0
    short_velocity: float = 0.0
    short_coherence: float = 0.0


class BNBMarketRetina:
    """Convert observed BNB price and volume history into biological visual motion."""

    def __init__(self, width: int = 32, height: int = 16) -> None:
        if width < 4 or height < 4:
            raise ValueError("retina dimensions are too small")
        self.width = width
        self.height = height

    def encode(
        self,
        prices: tuple[float, ...],
        volumes: tuple[float, ...] | None = None,
    ) -> RetinaStimulus:
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

        # Volume-induced luminance / arousal contrast
        volume_contrast = 1.0
        if volumes is not None and len(volumes) >= 4:
            vol_window = volumes[-len(window) :]
            mean_vol = sum(vol_window) / max(1, len(vol_window))
            curr_vol = vol_window[-1] if vol_window else mean_vol
            # Contrast gain in [0.6, 1.6] mimicking photoreceptor gain adaptation
            if mean_vol > 1e-12:
                volume_contrast = max(0.60, min(1.60, float(curr_vol / mean_vol)))

        for i, row in enumerate(positions):
            x = start + i
            if i == 0:
                on[row][x] = 0.5 * volume_contrast
                off[row][x] = 0.5 * volume_contrast
                continue
            delta = window[i] / window[i - 1] - 1.0
            velocities.append(delta)
            strength = min(1.0, abs(delta) * 100.0) * volume_contrast
            previous = positions[i - 1]
            if delta > 0:
                up += strength
                on[row][x] = min(1.0, 1.0 * volume_contrast)
                on[previous][x] = max(on[previous][x], min(1.0, strength))
            elif delta < 0:
                down += strength
                off[row][x] = min(1.0, 1.0 * volume_contrast)
                off[previous][x] = max(off[previous][x], min(1.0, strength))
            else:
                on[row][x] = 0.5 * volume_contrast
                off[row][x] = 0.5 * volume_contrast

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

        # Multi-scale dynamics: Short window (last 4-6 moves)
        short_window_len = min(6, len(velocities))
        if short_window_len > 1:
            short_vels = velocities[-short_window_len:]
            short_mean_vel = sum(short_vels) / short_window_len
            short_mean_abs = sum(abs(v) for v in short_vels) / short_window_len
            short_velocity = max(-1.0, min(1.0, short_mean_vel * 100.0))
            short_coherence = max(0.0, min(1.0, abs(short_mean_vel) / max(short_mean_abs, 1e-12)))
        else:
            short_velocity = max(-1.0, min(1.0, mean_velocity * 100.0))
            short_coherence = max(0.0, min(1.0, coherence))

        # Local volatility contrast: standard deviation of returns relative to span
        volatility_contrast = min(2.0, (span / max(lo, 1e-12)) * 100.0)

        return RetinaStimulus(
            on_field=tuple(tuple(min(1.0, value) for value in row) for row in on),
            off_field=tuple(tuple(min(1.0, value) for value in row) for row in off),
            directions=directions,
            coherence=max(0.0, min(1.0, coherence)),
            velocity=max(-1.0, min(1.0, mean_velocity * 100.0)),
            acceleration=max(-1.0, min(1.0, acceleration * 100.0)),
            volume_contrast=volume_contrast,
            volatility_contrast=volatility_contrast,
            short_velocity=short_velocity,
            short_coherence=short_coherence,
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
