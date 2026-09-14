from flydeck.visual_agent import MaleCNSVisualSystem
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron
from flydeck.visual_diagnostics import (
    DIRECTIONS,
    make_motion_sequence,
    make_motion_stimulus,
    run_motion_diagnostic,
    run_motion_suite,
)


def _circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L1", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "L2", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(4, "L2", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(5, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(7, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(8, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=(
            VisualEdge(0, 4, 1.0),
            VisualEdge(2, 5, 1.0),
            VisualEdge(1, 6, 1.0),
            VisualEdge(3, 7, 1.0),
        ),
        l1_inputs=(0, 1),
        l2_inputs=(2, 3),
        t4_outputs=((), (), (4,), (5,)),
        t5_outputs=((), (), (6,), (7,)),
        spatial_mode="soma_xy_proxy",
    )


def test_controlled_stimulus_is_spatial_and_causal() -> None:
    stimulus = make_motion_stimulus("up", width=8, height=6, polarity="on")
    assert stimulus.coherence == 1.0
    assert stimulus.directions == (0.0, 0.0, 0.0, 0.0)
    assert sum(value for row in stimulus.on_field for value in row) > 0.0
    assert sum(value for row in stimulus.off_field for value in row) == 0.0


def test_motion_sequence_changes_position_over_time() -> None:
    up = make_motion_sequence("up", width=8, height=8, steps=5)
    down = make_motion_sequence("down", width=8, height=8, steps=5)
    right = make_motion_sequence("right", width=8, height=8, steps=5)
    left = make_motion_sequence("left", width=8, height=8, steps=5)

    up_rows = [next(y for y, row in enumerate(item.on_field) if any(row)) for item in up]
    down_rows = [next(y for y, row in enumerate(item.on_field) if any(row)) for item in down]
    right_cols = [next(x for x in range(8) if any(row[x] for row in item.on_field)) for item in right]
    left_cols = [next(x for x in range(8) if any(row[x] for row in item.on_field)) for item in left]

    assert up_rows[0] < up_rows[-1]
    assert down_rows[0] > down_rows[-1]
    assert right_cols[0] < right_cols[-1]
    assert left_cols[0] > left_cols[-1]
    assert len(set(up_rows)) > 1
    assert len(set(down_rows)) > 1
    assert len(set(right_cols)) > 1
    assert len(set(left_cols)) > 1


def test_spatial_entry_mapping_produces_local_drive() -> None:
    visual = MaleCNSVisualSystem(_circuit())
    stimulus = make_motion_stimulus("up", width=8, height=6, polarity="on", position=0.0)
    visual.step(stimulus)
    assert visual.last_entry_drive[0] != visual.last_entry_drive[1]
    assert visual.last_entry_drive[2] == 0.0
    assert visual.last_entry_drive[3] == 0.0


def test_neural_state_is_nonnegative() -> None:
    visual = MaleCNSVisualSystem(_circuit())
    sequence = make_motion_sequence("right", width=8, height=8, steps=6)
    for stimulus in sequence:
        state = visual.step(stimulus)
        assert all(value >= 0.0 for value in state)


def test_temporal_adaptation_adds_transient_drive() -> None:
    visual = MaleCNSVisualSystem(_circuit(), temporal_gain=1.0)
    first = make_motion_stimulus("right", width=8, height=8, position=0.2)
    second = make_motion_stimulus("right", width=8, height=8, position=0.8)
    visual.step(first)
    first_drive = tuple(visual.last_entry_drive)
    visual.step(second)
    second_drive = tuple(visual.last_entry_drive)
    assert max(second_drive) >= max(first_drive)
    assert any(second > first for first, second in zip(first_drive, second_drive))


def test_motion_diagnostic_returns_temporal_trace() -> None:
    result = run_motion_diagnostic(_circuit(), direction="down", polarity="off", steps=5)
    assert result.name == "OFF_DOWN"
    assert len(result.t4) == 4
    assert len(result.t5) == 4
    assert len(result.temporal_t4) == 5
    assert len(result.temporal_t5) == 5
    assert all(len(frame) == 4 for frame in result.temporal_t4)
    assert result.up_score >= 0.0
    assert result.down_score >= 0.0


def test_motion_suite_covers_four_directions_and_polarities() -> None:
    results = run_motion_suite(_circuit(), steps=5)
    assert [item.name for item in results] == [
        "ON_RIGHT", "ON_LEFT", "ON_UP", "ON_DOWN",
        "OFF_RIGHT", "OFF_LEFT", "OFF_UP", "OFF_DOWN",
    ]
    assert all(item.direction in DIRECTIONS for item in results)
