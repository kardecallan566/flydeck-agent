"""Phase 6 integration tests: Internal State, Mushroom Body, Predictive Coding, Decision Engine."""
from __future__ import annotations

import pytest

from flydeck.internal_state import AgentInternalState
from flydeck.mushroom_body import MushroomBodyAssociativeMemory, MushroomBodyOutput
from flydeck.predictive_coding import PredictiveCodingEngine, PredictiveCodingUpdate
from flydeck.decision_engine import DynamicDecisionEngine, DecisionReason
from flydeck.neural_diagnostics import NeuralDiagnosticsTracer


# ── Internal State ────────────────────────────────────────────────────────────


def test_internal_state_copy_is_independent() -> None:
    """Copy must be a separate instance so mutations don't propagate."""
    state = AgentInternalState()
    state.arousal = 0.80
    state.cx_heading = -0.50
    clone = state.copy()
    assert clone.arousal == 0.80
    assert clone.cx_heading == -0.50
    state.arousal = 0.10
    assert clone.arousal == 0.80  # independent


def test_internal_state_diagnostic_dict() -> None:
    state = AgentInternalState()
    state.vs_net = 0.35
    d = state.to_diagnostic_dict()
    assert "vs_net" in d
    assert d["vs_net"] == 0.35
    assert "p_up" in d and "p_down" in d and "p_chop" in d


# ── Mushroom Body ─────────────────────────────────────────────────────────────


def test_mb_sparsity_constraint() -> None:
    """Kenyon Cell activation must be strictly sparse (~5%)."""
    mb = MushroomBodyAssociativeMemory(input_dim=8, kc_count=256, sparsity_fraction=0.05)
    context = (0.5, -0.3, 0.1, 0.8, -0.2, 0.4, 0.0, -0.6)
    out = mb.perceive(context)
    active_fraction = len(out.active_kc_indices) / 256
    assert 0.03 <= active_fraction <= 0.08, f"Sparsity violated: {active_fraction:.2%}"


def test_mb_causal_reinforcement_no_leakage() -> None:
    """Learning must only occur AFTER reinforce() is called (t+1 causality)."""
    mb = MushroomBodyAssociativeMemory(input_dim=8, kc_count=256, learning_rate=0.10)
    context = (0.5, -0.3, 0.1, 0.8, -0.2, 0.4, 0.0, -0.6)

    # First perceive: no prior learning, valence should be ~0
    out1 = mb.perceive(context)
    assert abs(out1.valence) < 0.01, "Valence should start near zero"

    # Reinforce with strong positive signal (simulating UP confirmed)
    mb.reinforce(observed_return=0.05)

    # Re-perceive same context: should now have positive valence
    out2 = mb.perceive(context)
    assert out2.valence > 0.0, "Valence should be positive after UP reinforcement"


def test_mb_novelty_decreases_with_repetition() -> None:
    """Repeated contexts should become familiar (lower novelty)."""
    mb = MushroomBodyAssociativeMemory(input_dim=8, kc_count=256)
    context = (0.5, -0.3, 0.1, 0.8, -0.2, 0.4, 0.0, -0.6)

    out1 = mb.perceive(context)
    mb.reinforce(0.01)
    out2 = mb.perceive(context)
    mb.reinforce(0.01)
    out3 = mb.perceive(context)

    assert out3.novelty <= out1.novelty, "Novelty should decrease with familiarity"


def test_mb_disabled_returns_zero_valence() -> None:
    mb = MushroomBodyAssociativeMemory(enabled=False)
    out = mb.perceive((0.5, -0.3, 0.1, 0.8, -0.2, 0.4, 0.0, -0.6))
    assert out.valence == 0.0
    assert out.active_kc_indices == ()


# ── Predictive Coding ─────────────────────────────────────────────────────────


def test_predictive_coding_surprise_on_reversal() -> None:
    """A sudden market reversal should produce high prediction error (surprise)."""
    pc = PredictiveCodingEngine(expectation_learning_rate=0.30, surprise_threshold=0.20)

    # Feed consistent upward motion to build UP expectation
    for _ in range(5):
        update = pc.step(observed_motion=0.60, cx_context=0.40, mb_valence=0.10, volatility=0.003)

    # Sudden reversal: strong downward
    surprise_update = pc.step(observed_motion=-0.70, cx_context=0.40, mb_valence=0.10, volatility=0.003)
    assert surprise_update.prediction_error > 0.30, "Should detect strong surprise"
    assert surprise_update.dopamine_burst > 0.0, "Should fire dopaminergic burst"


def test_predictive_coding_consistency_builds_confidence() -> None:
    """Consistent directional motion should reduce uncertainty over time."""
    pc = PredictiveCodingEngine()

    uncertainties = []
    for i in range(8):
        update = pc.step(observed_motion=0.50, cx_context=0.30, mb_valence=0.20, volatility=0.002)
        uncertainties.append(update.uncertainty)

    # After consistent UP signals, uncertainty should decrease
    assert uncertainties[-1] < uncertainties[0], "Uncertainty should decrease with consistency"


def test_predictive_coding_hypothesis_probs_sum_to_one() -> None:
    pc = PredictiveCodingEngine()
    update = pc.step(observed_motion=0.40, cx_context=0.20, mb_valence=0.10, volatility=0.003)
    total = sum(update.hypothesis_probs)
    assert abs(total - 1.0) < 1e-6, f"Hypothesis probs must sum to 1, got {total}"


# ── Decision Engine ──────────────────────────────────────────────────────────


def test_decision_engine_wait_on_high_conflict() -> None:
    """Opposing signals from different circuits should trigger WAIT."""
    engine = DynamicDecisionEngine(max_conflict_tolerance=0.30)
    state = AgentInternalState()
    state.vs_net = 0.50      # LPTC says UP
    state.retina_velocity = -0.50  # Retina says DOWN
    state.cx_heading = 0.30  # CX says UP
    state.coherence = 0.30
    state.arousal = 0.15
    state.hypothesis_probs = (0.45, 0.35, 0.20)

    decision = engine.decide(state, minimum_confidence=0.10)
    assert decision.wait, "Should WAIT when LPTC and Retina disagree strongly"
    assert decision.reason in (DecisionReason.WAIT_HIGH_CONFLICT, DecisionReason.WAIT_LOW_CONFIDENCE)


def test_decision_engine_commits_on_consensus() -> None:
    """Strong aligned signals should produce a directional commitment."""
    engine = DynamicDecisionEngine(max_conflict_tolerance=0.40)
    state = AgentInternalState()
    state.vs_net = 0.60
    state.retina_velocity = 0.55
    state.cx_heading = 0.40
    state.mb_valence = 0.30
    state.coherence = 0.70
    state.arousal = 0.15
    state.uncertainty = 0.30
    state.hypothesis_probs = (0.65, 0.15, 0.20)

    decision = engine.decide(state, minimum_confidence=0.10)
    assert not decision.wait, "Should commit when all circuits agree"
    assert decision.reason == DecisionReason.COMMIT_UP


def test_decision_engine_wait_on_high_uncertainty() -> None:
    engine = DynamicDecisionEngine(max_uncertainty_tolerance=0.80)
    state = AgentInternalState()
    state.vs_net = 0.10
    state.retina_velocity = 0.05
    state.cx_heading = 0.02
    state.mb_valence = -0.03
    state.coherence = 0.50
    state.arousal = 0.15
    state.uncertainty = 0.95
    state.hypothesis_probs = (0.34, 0.33, 0.33)

    decision = engine.decide(state, minimum_confidence=0.10)
    assert decision.wait, "Should WAIT when uncertainty is very high"


# ── Neural Diagnostics ───────────────────────────────────────────────────────


def test_diagnostics_records_and_formats() -> None:
    tracer = NeuralDiagnosticsTracer()
    state = AgentInternalState()
    state.vs_net = 0.40
    state.arousal = 0.25

    from flydeck.decision_engine import DecoupledDecision
    from flydeck.bnb_prediction import Prediction

    decision = DecoupledDecision(
        action=Prediction.UP,
        reason=DecisionReason.COMMIT_UP,
        up_score=0.45,
        down_score=0.10,
        confidence=0.35,
        wait=False,
        conflict=0.05,
        uncertainty=0.25,
        temporal_consistency=0.90,
    )
    entry = tracer.record(round_index=42, close_price=600.0, state=state, decision=decision)
    assert entry.round_index == 42
    assert entry.action == "UP"
    assert len(tracer.entries) == 1

    table = tracer.summary_table(last_n=5)
    assert "42" in table
    assert "COMMIT_UP" in table
