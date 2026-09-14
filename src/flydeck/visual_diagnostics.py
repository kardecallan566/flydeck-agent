from __future__ import annotations

from dataclasses import dataclass

from .market_retina import RetinaStimulus
from .visual_agent import MaleCNSVisualSystem
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class MotionDiagnostic:
    name: str
    t4: tuple[float, float, float, float]
    t5: tuple[float, float, float, float]
    up_score: float
    down_score: float
    confidence: float


def make_motion_stimulus(
    direction: str,
    *,
    width: int = 32,
    height: int = 16,
    polarity: str = "on",
    position: float = 0.5,
) -> RetinaStimulus:
    """Create one controlled visual edge frame for circuit validation.

    This is a laboratory stimulus, not a financial feature. The actual visual
    input is the ON/OFF field; the metadata channels are deliberately neutral
    so the circuit cannot receive the answer through a direction feature.
    """
    if width < 4 or height < 4:
        raise ValueError("stimulus dimensions are too small")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if polarity not in {"on", "off"}:
        raise ValueError("polarity must be 'on' or 'off'")
    if not 0.0 <= position <= 1.0:
        raise ValueError("position must be between 0 and 1")

    y = int(round(position * (height - 1)))
    on = [[0.0] * width for _ in range(height)]
    off = [[0.0] * width for _ in range(height)]
    field = on if polarity == "on" else off

    # A horizontal luminance edge is the object. Direction is represented by
    # its displacement between consecutive frames, not by this frame itself.
    for x in range(width):
        field[y][x] = 1.0

    vertical = 1.0 if direction == "up" else -1.0
    return RetinaStimulus(
        on_field=tuple(tuple(row) for row in on),
        off_field=tuple(tuple(row) for row in off),
        directions=(0.0, 0.0, 0.0, 0.0),
        coherence=1.0,
        velocity=vertical,
        acceleration=0.0,
    )


def make_motion_sequence(
    direction: str,
    *,
    width: int = 32,
    height: int = 16,
    polarity: str = "on",
    steps: int = 12,
    start: float = 0.15,
    end: float = 0.85,
) -> tuple[RetinaStimulus, ...]:
    """Create a causal sequence where the same edge moves across the field."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("start and end must be between 0 and 1")

    if direction == "down":
        start, end = end, start

    positions = [
        start + (end - start) * index / (steps - 1)
        for index in range(steps)
    ]
    return tuple(
        make_motion_stimulus(
            direction,
            width=width,
            height=height,
            polarity=polarity,
            position=position,
        )
        for position in positions
    )


def run_motion_diagnostic(
    circuit: VisualCircuit,
    *,
    direction: str,
    polarity: str = "on",
    steps: int = 12,
) -> MotionDiagnostic:
    """Run one causal motion sequence through the real extracted circuit."""
    sequence = make_motion_sequence(direction, polarity=polarity, steps=steps)
    visual = MaleCNSVisualSystem(circuit)
    for stimulus in sequence:
        visual.step(stimulus)
    decision = visual.decision(minimum_confidence=0.0)
    t4 = visual._directional_activity(circuit.t4_outputs)
    t5 = visual._directional_activity(circuit.t5_outputs)
    return MotionDiagnostic(
        name=f"{polarity.upper()}_{direction.upper()}",
        t4=t4,
        t5=t5,
        up_score=decision.up_score,
        down_score=decision.down_score,
        confidence=decision.confidence,
    )


def run_motion_suite(circuit: VisualCircuit, *, steps: int = 12) -> tuple[MotionDiagnostic, ...]:
    """Run ON/OFF upward/downward controls without using market labels."""
    return tuple(
        run_motion_diagnostic(circuit, direction=direction, polarity=polarity, steps=steps)
        for polarity in ("on", "off")
        for direction in ("up", "down")
    )


def format_diagnostic(diagnostic: MotionDiagnostic) -> str:
    return (
        f"{diagnostic.name}: "
        f"T4={_fmt4(diagnostic.t4)} "
        f"T5={_fmt4(diagnostic.t5)} "
        f"UP={diagnostic.up_score:.6f} "
        f"DOWN={diagnostic.down_score:.6f} "
        f"confidence={diagnostic.confidence:.3f}"
    )


def _fmt4(values: tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"
