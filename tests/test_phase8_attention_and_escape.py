"""Unit tests for Phase 8: Attention, Giant Fiber Escape, Metabolic Control and CX-MB feedback."""
from pathlib import Path
import pytest

from flydeck.attention_system import TopDownAttentionModule
from flydeck.checkpoint import AgentCheckpointManager
from flydeck.decision_engine import DecisionReason, DynamicDecisionEngine
from flydeck.giant_fiber import GiantFiberEscapeCircuit
from flydeck.internal_state import AgentInternalState
from flydeck.metabolic_control import MetabolicRiskController
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


# ── 1. Top-Down Attention ───────────────────────────────────────────────────


def test_top_down_attention_focus_increases_under_arousal() -> None:
    att = TopDownAttentionModule()
    # Calm state
    calm = att.step(arousal=0.10, prediction_error=0.05, uncertainty=0.20, volatility=0.002)
    # High arousal + surprise state
    agitated = att.step(arousal=0.85, prediction_error=0.70, uncertainty=0.30, volatility=0.008)
    assert agitated.temporal_focus > calm.temporal_focus, "Attention should focus on recent candles during surprise"


# ── 2. Giant Fiber Escape Circuit ───────────────────────────────────────────


def test_giant_fiber_triggers_on_looming_shock() -> None:
    gf = GiantFiberEscapeCircuit(shock_threshold=0.60, cooldown_rounds=2)
    # Normal moderate candle
    normal = gf.step(current_return_pct=0.20, prediction_error=0.15, volatility_contrast=0.003, volume_contrast=1.1)
    assert not normal.is_shock

    # Violent looming shock: 2% move + high surprise + volume surge
    shock = gf.step(current_return_pct=2.10, prediction_error=1.10, volatility_contrast=0.015, volume_contrast=3.5)
    assert shock.is_shock, "Giant Fiber must fire on looming market anomaly"
    assert shock.cooldown_remaining == 2

    # Next round should still maintain cooldown protection
    next_step = gf.step(current_return_pct=0.10, prediction_error=0.10, volatility_contrast=0.003, volume_contrast=1.0)
    assert next_step.is_shock, "Cooldown should maintain protection"
    assert next_step.cooldown_remaining == 1


def test_decision_engine_enforces_wait_regime_shock() -> None:
    engine = DynamicDecisionEngine()
    state = AgentInternalState()
    state.is_shock = True  # Giant Fiber emergency active
    state.vs_net = 0.80    # Even with huge directional signal
    state.hypothesis_probs = (0.80, 0.10, 0.10)

    decision = engine.decide(state, minimum_confidence=0.10)
    assert decision.wait
    assert decision.reason == DecisionReason.WAIT_REGIME_SHOCK, "Must force WAIT_REGIME_SHOCK during emergency"


# ── 3. Metabolic Risk Controller ────────────────────────────────────────────


def test_metabolic_controller_modulates_threshold() -> None:
    ctrl = MetabolicRiskController(base_energy=1.0)
    # Consecutive positive rewards feed the agent
    for _ in range(5):
        ctrl.update_feedback(realized_return=0.80)
    fed_state = ctrl.step()
    assert fed_state.energy_level > 1.0
    assert fed_state.threshold_modifier < 0.0, "High energy should slightly relax threshold"

    # Consecutive heavy losses deplete reserves
    for _ in range(10):
        ctrl.update_feedback(realized_return=-1.50)
    starved_state = ctrl.step()
    assert starved_state.energy_level < 1.0
    assert starved_state.threshold_modifier > 0.0, "Low energy must elevate threshold for defense"


# ── 4. Recurrent CX-MB Feedback & Checkpoint Round-Trip ─────────────────────


def test_phase8_checkpoint_round_trip(tmp_path: Path) -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)

    agent.perceive((100.0, 101.0, 102.0, 103.0))
    agent.visual.attention._temporal_focus = 1.45
    agent.visual.giant_fiber._cooldown = 2
    agent.visual.metabolic_control._energy = 0.82

    ckpt_file = tmp_path / "phase8_brain.json"
    AgentCheckpointManager.save(agent, ckpt_file)

    fresh_agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=4)
    AgentCheckpointManager.load(fresh_agent, ckpt_file)

    assert abs(fresh_agent.visual.attention._temporal_focus - 1.45) < 1e-4
    assert fresh_agent.visual.giant_fiber._cooldown == 2
    assert abs(fresh_agent.visual.metabolic_control._energy - 0.82) < 1e-4
