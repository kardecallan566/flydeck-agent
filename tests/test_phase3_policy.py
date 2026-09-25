from flydeck.bnb_prediction import Prediction
from flydeck.eligibility import SparseEligibilityTrace
from flydeck.episodic_memory import BoundedEpisodicMemory
from flydeck.risk_policy import LightweightRiskPolicy


def test_sparse_eligibility_decays_and_returns_reward_updates() -> None:
    trace = SparseEligibilityTrace(decay=0.5)
    trace.step({1: 1.0, 3: -0.5})
    updates = trace.reinforce(1.0, learning_rate=0.1)
    assert updates[1] > 0.0
    trace.step({})
    assert abs(trace.values[1]) < 1.0


def test_episodic_memory_is_bounded_and_reports_novelty() -> None:
    memory = BoundedEpisodicMemory(capacity=8)
    for index in range(20):
        memory.add((float(index), 0.0), Prediction.UP, 1.0, "TREND_UP")
    assert memory.size == 8
    assert memory.novelty((19.0, 0.0)) < memory.novelty((100.0, 0.0))
    assert memory.retrieve((19.0, 0.0), "TREND_UP")


def test_risk_policy_cooldown_is_separate_from_neural_learning() -> None:
    policy = LightweightRiskPolicy(max_consecutive_losses=2, cooldown_rounds=2)
    policy.observe(Prediction.UP, -1.0)
    policy.observe(Prediction.UP, -1.0)
    assert policy.before_action(Prediction.DOWN) == Prediction.WAIT
    assert policy.state.cooldown > 0
