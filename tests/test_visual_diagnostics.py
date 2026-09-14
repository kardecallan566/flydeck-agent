from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron
from flydeck.visual_diagnostics import make_motion_stimulus, run_motion_diagnostic, run_motion_suite


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
        edges=(VisualEdge(0, 4, 1.0), VisualEdge(2, 5, 1.0), VisualEdge(1, 6, 1.0), VisualEdge(3, 7, 1.0)),
        l1_inputs=(0, 1),
        l2_inputs=(2, 3),
        t4_outputs=((), (), (4,), (5,)),
        t5_outputs=((), (), (6,), (7,)),
        spatial_mode="soma_xy_proxy",
    )


def test_controlled_stimulus_is_spatial_and_causal() -> None:
    stimulus = make_motion_stimulus("up", width=8, height=6, polarity="on")
    assert stimulus.coherence == 1.0
    assert stimulus.directions[2] == 1.0
    assert sum(value for row in stimulus.on_field for value in row) > 0.0
    assert sum(value for row in stimulus.off_field for value in row) == 0.0


def test_spatial_entry_mapping_produces_local_drive() -> None:
    visual = __import__("flydeck.visual_agent", fromlist=["MaleCNSVisualSystem"]).MaleCNSVisualSystem(_circuit())
    stimulus = make_motion_stimulus("up", width=8, height=6, polarity="on", position=0.0)
    visual.step(stimulus)
    assert visual.last_entry_drive[0] != visual.last_entry_drive[1]
    assert visual.last_entry_drive[2] == 0.0
    assert visual.last_entry_drive[3] == 0.0


def test_motion_diagnostic_returns_all_directional_groups() -> None:
    result = run_motion_diagnostic(_circuit(), direction="down", polarity="off")
    assert result.name == "OFF_DOWN"
    assert len(result.t4) == 4
    assert len(result.t5) == 4
    assert result.up_score >= 0.0
    assert result.down_score >= 0.0


def test_motion_suite_contains_on_off_up_down_controls() -> None:
    results = run_motion_suite(_circuit(), steps=2)
    assert [item.name for item in results] == ["ON_UP", "ON_DOWN", "OFF_UP", "OFF_DOWN"]
