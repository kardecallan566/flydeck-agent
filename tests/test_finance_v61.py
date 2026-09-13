from pytest import approx

from flydeck.agent import Agent
from flydeck.finance_training import _gate_decision, _choose_finance_action


def test_adaptive_gate_relaxes_when_scores_are_small():
    gated, confidence, adaptive_margin = _gate_decision(
        (0.00, 0.03, -0.01), opportunity_margin=0.08, confidence_threshold=0.20
    )
    assert adaptive_margin == approx(0.01)
    assert confidence == approx(0.75)
    assert not gated


def test_adaptive_gate_still_blocks_weak_trade_opportunity():
    gated, confidence, adaptive_margin = _gate_decision(
        (0.25, 0.26, 0.24),
        opportunity_margin=0.08,
        confidence_threshold=0.20,
    )
    assert adaptive_margin == approx(0.005)
    assert confidence == approx(0.50)
    assert not gated


def test_adaptive_gate_blocks_when_hold_is_best():
    gated, confidence, _adaptive_margin = _gate_decision(
        (0.30, 0.20, 0.10),
        opportunity_margin=0.08,
        confidence_threshold=0.20,
    )
    assert gated
    assert confidence == 0.0


def test_choose_action_preserves_network_direction_when_gate_opens():
    agent = Agent(observation_size=3, action_size=3, hidden_size=8, density=0.5, seed=7)
    action, gated, confidence = _choose_finance_action(
        agent, (0.20, 0.50, 0.10), opportunity_margin=0.08, confidence_threshold=0.20
    )
    assert action == 1
    assert not gated
    assert confidence > 0.20
