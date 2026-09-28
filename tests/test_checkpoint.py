"""Tests for AgentCheckpointManager."""
from pathlib import Path
import pytest

from flydeck.bnb_prediction import Prediction
from flydeck.checkpoint import AgentCheckpointManager
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit, VisualNeuron, VisualEdge


def _toy_circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.2, 0.2),
        VisualNeuron(2, "L2", "visual_entry", "gaba", -1.0, 0.8, 0.8),
        VisualNeuron(3, "T4", "motion_t4", "acetylcholine", 1.0, 0.2, 0.5),
        VisualNeuron(4, "T4", "motion_t4", "acetylcholine", 1.0, 0.2, 0.5),
        VisualNeuron(5, "T5", "motion_t5", "acetylcholine", 1.0, 0.8, 0.5),
        VisualNeuron(6, "T5", "motion_t5", "acetylcholine", 1.0, 0.8, 0.5),
    )
    edges = (
        VisualEdge(0, 2, 1.0),
        VisualEdge(1, 4, 1.0),
        VisualEdge(2, 3, 0.5),
        VisualEdge(4, 5, 0.5),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=edges,
        l1_inputs=(0,),
        l2_inputs=(1,),
        t4_outputs=((), (), (2,), (3,)),
        t5_outputs=((), (), (4,), (5,)),
        spatial_mode="soma_xy_proxy",
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)

    # Simulate a couple steps to build state
    agent.perceive((100.0, 101.0, 102.0, 103.0))
    agent.perceive((103.0, 102.5, 104.0, 105.0))

    # Manually tweak some learned weights to verify exact restoration
    agent.visual.mushroom_body._mbon_weights_np[10] = 1.234
    agent.visual.central_complex._attractor_heading = 0.567
    agent.visual.predictive_coding._current_expectation = -0.432
    agent.visual._previous_price = 105.0
    agent.visual.policy_bias = 0.075
    agent.visual._previous_action_sign = 1
    agent.visual._previous_policy_action = Prediction.UP
    agent.visual._previous_policy_vector = (0.1, -0.2, 0.3)
    agent.visual._previous_policy_regime = "TREND_UP"
    agent.visual.eligibility._values = {3: 0.75, 9: -0.25}
    agent.visual.risk_policy.observe(Prediction.DOWN, -1.0)
    agent.visual.temporal_memory.update(0.4, novelty=0.2)
    agent.visual.episodic_memory.add((0.1, 0.2), Prediction.UP, 1.0, "TREND_UP")
    agent.visual.internal_state.arousal = 0.77
    agent.visual.internal_state.hypothesis_probs = (0.6, 0.3, 0.1)

    ckpt_file = tmp_path / "fly_brain.json"
    metadata = {"round": 42, "pnl": 12.5}
    saved_path = AgentCheckpointManager.save(agent, ckpt_file, metadata=metadata)
    assert saved_path.exists()

    # Create fresh agent with zeroed state
    fresh_agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)
    assert fresh_agent.visual.mushroom_body._mbon_weights_np[10] == 0.0

    # Load checkpoint
    loaded_meta = AgentCheckpointManager.load(fresh_agent, ckpt_file)
    assert loaded_meta["round"] == 42
    assert loaded_meta["pnl"] == 12.5

    assert abs(fresh_agent.visual.mushroom_body._mbon_weights_np[10] - 1.234) < 1e-5
    assert abs(fresh_agent.visual.central_complex._attractor_heading - 0.567) < 1e-5
    assert abs(fresh_agent.visual.predictive_coding._current_expectation - (-0.432)) < 1e-5
    assert fresh_agent.visual._previous_price == 105.0
    assert abs(fresh_agent.visual.policy_bias - 0.075) < 1e-9
    assert fresh_agent.visual._previous_action_sign == 1
    assert fresh_agent.visual._previous_policy_action == Prediction.UP
    assert fresh_agent.visual._previous_policy_vector == (0.1, -0.2, 0.3)
    assert fresh_agent.visual._previous_policy_regime == "TREND_UP"
    assert fresh_agent.visual.eligibility.values == {3: 0.75, 9: -0.25}
    assert fresh_agent.visual.risk_policy.state.consecutive_losses == 1
    assert abs(fresh_agent.visual.temporal_memory.state.fast - agent.visual.temporal_memory.state.fast) < 1e-9
    assert fresh_agent.visual.episodic_memory.size == 1
    assert abs(fresh_agent.visual.internal_state.arousal - 0.77) < 1e-9
    assert fresh_agent.visual.internal_state.hypothesis_probs == (0.6, 0.3, 0.1)
