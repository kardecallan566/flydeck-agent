import numpy as np

from flydeck.mushroom_body import MushroomBodyAssociativeMemory
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron


def circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=(VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)),
        l1_inputs=(0,),
        l2_inputs=(1,),
        t4_outputs=((), (), (2,), (3,)),
        t5_outputs=((), (), (4,), (5,)),
        spatial_mode="soma_xy_proxy",
    )


def test_mushroom_body_updates_three_action_readouts() -> None:
    mb = MushroomBodyAssociativeMemory(input_dim=2, kc_count=32)
    mb.perceive((1.0, 0.0))
    before = mb._action_weights_np.copy()
    mb.reinforce_actions((0.2, 1.0, -1.0))
    assert not np.array_equal(before[0], mb._action_weights_np[0])
    assert not np.array_equal(before[1], mb._action_weights_np[1])
    assert not np.array_equal(before[2], mb._action_weights_np[2])


def test_visual_agent_can_freeze_learning() -> None:
    agent = FlyVisualPredictionAgent(circuit(), retina_width=8, retina_height=6, confidence_threshold=0.0)
    agent.perceive((100.0, 100.0, 100.5, 101.0, 101.5))
    agent.set_learning(False)
    before = agent.visual.mushroom_body._action_weights_np.copy()
    agent.perceive((101.5, 101.0, 100.5, 100.0, 99.5))
    assert np.array_equal(before, agent.visual.mushroom_body._action_weights_np)
