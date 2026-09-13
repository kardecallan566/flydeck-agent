from pytest import approx

from flydeck.agent import Agent
from flydeck.finance_v7 import gate_action, train_synthetic_crypto_v7


def test_v7_gate_opens_for_clear_small_score_opportunity():
    gate = gate_action((0.00, 0.03, -0.01), opportunity_margin=0.08, confidence_threshold=0.55)
    assert gate.adaptive_margin == approx(0.01)
    assert gate.confidence == approx(0.7361247, rel=1e-5)
    assert gate.opportunity
    assert not gate.gated
    assert gate.action == 1


def test_v7_gate_blocks_balanced_trade_scores():
    gate = gate_action((0.25, 0.26, 0.24), opportunity_margin=0.08, confidence_threshold=0.55)
    assert gate.adaptive_margin == approx(0.005)
    assert gate.confidence == approx(0.5064804, rel=1e-5)
    assert gate.gated
    assert not gate.opportunity
    assert gate.action == 0


def test_v7_gate_blocks_when_hold_is_best():
    gate = gate_action((0.30, 0.20, 0.10), opportunity_margin=0.08, confidence_threshold=0.55)
    assert gate.gated
    assert gate.confidence == 0.0
    assert not gate.opportunity


def test_v7_training_runs_with_raw_and_execution_layers():
    agent = Agent(observation_size=27, action_size=3, hidden_size=32, density=0.10, learning_rate=0.005, seed=42)
    result = train_synthetic_crypto_v7(
        agent,
        episodes=3,
        market_length=64,
        max_steps=30,
        seed=100,
        epsilon=0.20,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    )
    assert result.raw_total_actions == result.executed_total_actions
    assert result.total_trades >= 0
    assert 0.0 <= result.average_confidence <= 1.0
    assert result.average_prediction_error >= 0.0
    assert 0.0 <= result.opportunity_rate <= 1.0
