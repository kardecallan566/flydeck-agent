from flydeck.receptive_fields import infer_receptive_fields
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron


def _circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L1", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "Mi1", "motion_interneuron", "acetylcholine", 1.0),
        VisualNeuron(4, "Mi9", "motion_interneuron", "glutamate", -1.0),
        VisualNeuron(5, "T4c", "motion_detector", "acetylcholine", 1.0),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=(
            VisualEdge(0, 2, 1.0),
            VisualEdge(1, 3, 1.0),
            VisualEdge(2, 4, 1.0),
            VisualEdge(3, 4, 1.0),
        ),
        l1_inputs=(0, 1),
        l2_inputs=(),
        t4_outputs=((), (), (4,), ()),
        t5_outputs=((), (), (), ()),
        spatial_mode="soma_xy_proxy",
    )


def test_receptive_field_propagates_from_l1() -> None:
    fields = infer_receptive_fields(_circuit(), iterations=4)
    assert fields[2].x == 0.0
    assert fields[3].x == 1.0
    assert 0.0 < fields[4].x < 1.0
    assert fields[4].excitatory_x == 0.0
    assert fields[4].inhibitory_x == 1.0


def test_receptive_field_preserves_excitation_inhibition_offset() -> None:
    fields = infer_receptive_fields(_circuit(), iterations=4)
    offset = fields[4].excitation_inhibition_offset
    assert offset is not None
    assert offset[0] == 1.0
    assert offset[1] == 0.0
