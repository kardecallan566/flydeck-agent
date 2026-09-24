from flydeck.bnb_prediction import Prediction
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.mushroom_body import MushroomBodyAssociativeMemory
from flydeck.survival_training import train_visual_survival
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


def dataset(length: int = 48) -> BNBPredictionDataset:
    closes = tuple(100.0 + (1.0 if i % 2 == 0 else -0.7) * i for i in range(length))
    return BNBPredictionDataset(
        timestamps=tuple(range(length)),
        opens=closes,
        highs=tuple(v + 0.2 for v in closes),
        lows=tuple(v - 0.2 for v in closes),
        closes=closes,
        volumes=(10.0,) * length,
    )


def test_directional_feedback_corrects_wrong_down_toward_up() -> None:
    agent = FlyVisualPredictionAgent(circuit(), retina_width=8, retina_height=6, confidence_threshold=0.0)
    agent.perceive((100.0, 100.0, 100.0, 100.0, 100.0))
    agent.visual.register_action(Prediction.DOWN)
    agent.perceive((100.0, 100.5, 101.0, 101.5, 102.0))
    assert agent.visual.policy_bias > 0.0


def test_mushroom_body_death_reset_preserves_learned_weights() -> None:
    mb = MushroomBodyAssociativeMemory(input_dim=2, kc_count=32)
    mb.perceive((1.0, 0.0))
    mb.reinforce(1.0)
    learned = mb._mbon_weights_np.copy()
    mb.reset(preserve_weights=True)
    assert (mb._mbon_weights_np == learned).all()


def test_survival_training_is_causal_and_tracks_lives() -> None:
    _agent, result = train_visual_survival(
        dataset(), circuit(), context=8, initial_lives=2, max_rounds=20, confidence_threshold=0.0
    )
    assert result.rounds == 20
    assert result.lives_started == 2
    assert result.deaths >= 0
    assert result.up + result.down + result.wait == result.rounds
    assert 0.0 <= result.survival_rate <= 1.0
