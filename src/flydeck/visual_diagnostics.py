from __future__ import annotations

from dataclasses import dataclass

from .market_retina import RetinaStimulus
from .visual_agent import MaleCNSVisualSystem
from .visual_circuit import VisualCircuit

DIRECTIONS = ("right", "left", "up", "down")
DIRECTION_INDEX = {name: index for index, name in enumerate(DIRECTIONS)}


@dataclass(frozen=True, slots=True)
class MotionDiagnostic:
    name: str
    direction: str
    t4: tuple[float, float, float, float]
    t5: tuple[float, float, float, float]
    up_score: float
    down_score: float
    confidence: float
    temporal_t4: tuple[tuple[float, float, float, float], ...]
    temporal_t5: tuple[tuple[float, float, float, float], ...]

    @property
    def final_directional(self) -> tuple[float, float, float, float]:
        return tuple((a + b) / 2.0 for a, b in zip(self.t4, self.t5))

    @property
    def preferred_index(self) -> int:
        return DIRECTION_INDEX[self.direction]

    @property
    def opposite_index(self) -> int:
        if self.direction == "right":
            return 1
        if self.direction == "left":
            return 0
        if self.direction == "up":
            return 3
        return 2

    @property
    def final_preferred_score(self) -> float:
        return self.final_directional[self.preferred_index]

    @property
    def final_opposite_score(self) -> float:
        return self.final_directional[self.opposite_index]

    @property
    def selectivity(self) -> float:
        total = abs(self.final_preferred_score) + abs(self.final_opposite_score)
        if total <= 1e-12:
            return 0.0
        return (self.final_preferred_score - self.final_opposite_score) / total


def make_motion_stimulus(
    direction: str,
    *,
    width: int = 32,
    height: int = 16,
    polarity: str = "on",
    position: float = 0.5,
) -> RetinaStimulus:
    """Create one controlled visual edge frame for circuit validation.

    Direction is deliberately encoded only by the displacement of the edge
    across a sequence of frames. Direction metadata remains neutral.
    """
    if width < 4 or height < 4:
        raise ValueError("stimulus dimensions are too small")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if polarity not in {"on", "off"}:
        raise ValueError("polarity must be 'on' or 'off'")
    if not 0.0 <= position <= 1.0:
        raise ValueError("position must be between 0 and 1")

    coordinate = int(round(position * ((height if direction in {"up", "down"} else width) - 1)))
    on = [[0.0] * width for _ in range(height)]
    off = [[0.0] * width for _ in range(height)]
    field = on if polarity == "on" else off

    if direction in {"up", "down"}:
        for x in range(width):
            field[coordinate][x] = 1.0
    else:
        for y in range(height):
            field[y][coordinate] = 1.0

    return RetinaStimulus(
        on_field=tuple(tuple(row) for row in on),
        off_field=tuple(tuple(row) for row in off),
        directions=(0.0, 0.0, 0.0, 0.0),
        coherence=1.0,
        velocity=0.0,
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
    """Create a causal sequence where one edge moves across the field."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("start and end must be between 0 and 1")

    if direction in {"down", "left"}:
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
    visual: MaleCNSVisualSystem | None = None,
    fields: dict | None = None,
) -> MotionDiagnostic:
    """Run one causal motion sequence through the extracted circuit."""
    sequence = make_motion_sequence(direction, polarity=polarity, steps=steps)
    if visual is None:
        visual = MaleCNSVisualSystem(circuit, receptive_fields=fields)
    else:
        visual.reset()
    temporal_t4: list[tuple[float, float, float, float]] = []
    temporal_t5: list[tuple[float, float, float, float]] = []

    for stimulus in sequence:
        visual.step(stimulus)
        temporal_t4.append(visual._directional_activity(circuit.t4_outputs))
        temporal_t5.append(visual._directional_activity(circuit.t5_outputs))

    t4 = temporal_t4[-1]
    t5 = temporal_t5[-1]
    decision = visual.decision(minimum_confidence=0.0)
    return MotionDiagnostic(
        name=f"{polarity.upper()}_{direction.upper()}",
        direction=direction,
        t4=t4,
        t5=t5,
        up_score=decision.up_score,
        down_score=decision.down_score,
        confidence=decision.confidence,
        temporal_t4=tuple(temporal_t4),
        temporal_t5=tuple(temporal_t5),
    )


def run_motion_suite(
    circuit: VisualCircuit,
    *,
    steps: int = 12,
    fields: dict | None = None,
) -> tuple[MotionDiagnostic, ...]:
    """Run ON/OFF controls for all four motion directions."""
    visual = MaleCNSVisualSystem(circuit, receptive_fields=fields)
    return tuple(
        run_motion_diagnostic(circuit, direction=direction, polarity=polarity, steps=steps, visual=visual)
        for polarity in ("on", "off")
        for direction in DIRECTIONS
    )


def format_diagnostic(diagnostic: MotionDiagnostic) -> str:
    return (
        f"{diagnostic.name}: "
        f"T4={_fmt4(diagnostic.t4)} "
        f"T5={_fmt4(diagnostic.t5)} "
        f"UP={diagnostic.up_score:.6f} "
        f"DOWN={diagnostic.down_score:.6f} "
        f"confidence={diagnostic.confidence:.3f} "
        f"preferred={diagnostic.final_preferred_score:.6f} "
        f"opposite={diagnostic.final_opposite_score:.6f} "
        f"selectivity={diagnostic.selectivity:.3f}"
    )


def format_temporal_summary(diagnostic: MotionDiagnostic) -> str:
    preferred = diagnostic.preferred_index
    peak = max(
        abs(t4[preferred]) + abs(t5[preferred])
        for t4, t5 in zip(diagnostic.temporal_t4, diagnostic.temporal_t5)
    )
    final = abs(diagnostic.final_preferred_score)
    return (
        f"  temporal: frames={len(diagnostic.temporal_t4)} "
        f"final_preferred={final:.6f} peak_preferred={peak:.6f}"
    )


def _fmt4(values: tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"
