from flydeck.bnb_prediction import Prediction
from flydeck.decision_engine import DynamicDecisionEngine
from flydeck.internal_state import AgentInternalState


def test_disabled_decision_engine_removes_wait_gates() -> None:
    state = AgentInternalState()
    state.vs_net = 0.30
    state.retina_velocity = 0.25
    state.cx_heading = 0.10
    state.mb_valence = 0.05
    state.hypothesis_probs = (0.55, 0.20, 0.25)
    state.uncertainty = 1.0
    state.is_shock = True
    state.regime = "SHOCK"

    engine = DynamicDecisionEngine(enabled=False)
    decision = engine.decide(state)
    assert decision.action in (Prediction.UP, Prediction.DOWN)
    assert decision.wait is False
