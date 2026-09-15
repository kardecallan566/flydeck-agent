from flydeck.directional_mechanism import SpatialOffsetDirectionalMechanism
from flydeck.market_retina import RetinaStimulus
from flydeck.receptive_fields import ReceptiveField


def _field(x: float) -> RetinaStimulus:
    width = 9
    height = 5
    on = [[0.0] * width for _ in range(height)]
    column = min(width - 1, max(0, round(x * (width - 1))))
    for row in on:
        row[column] = 1.0
    return RetinaStimulus(
        on_field=tuple(tuple(row) for row in on),
        off_field=tuple(tuple(0.0 for _ in row) for row in on),
        directions=(0.0, 0.0, 0.0, 0.0),
        coherence=1.0,
        velocity=0.0,
        acceleration=0.0,
    )


def _run(positions: tuple[float, ...]) -> float:
    mechanism = SpatialOffsetDirectionalMechanism(
        {
            0: ReceptiveField(
                x=0.5,
                y=0.5,
                excitatory_x=0.25,
                excitatory_y=0.5,
                inhibitory_x=0.75,
                inhibitory_y=0.5,
                excitatory_mass=1.0,
                inhibitory_mass=1.0,
            )
        },
        inhibition_alpha=0.20,
    )
    responses = []
    for position in positions:
        responses.append(
            mechanism.step(_field(position), ((0,),), polarity="on")[0]
        )
    return sum(responses)


def test_offset_fast_excitation_slow_inhibition_prefers_excitation_first_motion() -> None:
    right = _run((0.10, 0.25, 0.40, 0.55, 0.70, 0.85))
    left = _run((0.90, 0.75, 0.60, 0.45, 0.30, 0.15))
    assert right > left


def test_directional_mechanism_does_not_use_direction_metadata() -> None:
    first = _field(0.25)
    second = _field(0.75)
    mechanism = SpatialOffsetDirectionalMechanism(
        {
            0: ReceptiveField(
                x=0.5,
                y=0.5,
                excitatory_x=0.25,
                excitatory_y=0.5,
                inhibitory_x=0.75,
                inhibitory_y=0.5,
                excitatory_mass=1.0,
                inhibitory_mass=1.0,
            )
        }
    )
    result_a = mechanism.step(first, ((0,),), polarity="on")[0]
    result_b = mechanism.step(second, ((0,),), polarity="on")[0]
    assert result_a > 0.0
    assert result_b >= 0.0
    assert first.directions == (0.0, 0.0, 0.0, 0.0)
    assert second.directions == (0.0, 0.0, 0.0, 0.0)
