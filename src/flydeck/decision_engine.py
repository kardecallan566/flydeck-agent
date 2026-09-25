"""Causal action selection with online centering and market-regime gating."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .bnb_prediction import Prediction
from .internal_state import AgentInternalState


class DecisionReason(str, Enum):
    COMMIT_UP = "COMMIT_UP"
    COMMIT_DOWN = "COMMIT_DOWN"
    WAIT_LOW_CONFIDENCE = "WAIT_LOW_CONFIDENCE"
    WAIT_HIGH_CONFLICT = "WAIT_HIGH_CONFLICT"
    WAIT_HIGH_UNCERTAINTY = "WAIT_HIGH_UNCERTAINTY"
    WAIT_NEUTRAL_REGIME = "WAIT_NEUTRAL_REGIME"
    WAIT_REGIME_SHOCK = "WAIT_REGIME_SHOCK"


@dataclass(frozen=True, slots=True)
class DecoupledDecision:
    action: Prediction
    reason: DecisionReason
    up_score: float
    down_score: float
    confidence: float
    wait: bool
    conflict: float
    uncertainty: float
    temporal_consistency: float
    p_wait: float = 0.0
    p_up: float = 0.5
    p_down: float = 0.5
    regime: str = "RANGE"


class DynamicDecisionEngine:
    """Select UP, DOWN, or WAIT from causal streams and regime context."""

    def __init__(self, base_confidence: float = 0.15, max_conflict_tolerance: float = 0.35,
                 max_uncertainty_tolerance: float = 0.88, history_window: int = 5,
                 enabled: bool = True) -> None:
        self.base_confidence = base_confidence
        self.max_conflict_tolerance = max_conflict_tolerance
        self.max_uncertainty_tolerance = max_uncertainty_tolerance
        self.history_window = history_window
        self.enabled = enabled
        self._recent_evidence_history: list[float] = []
        self._evidence_center = 0.0
        self._mb_center = 0.0
        self.center_learning_rate = 0.02
        self.wait_value_cap = 0.20
        self.wait_decision_margin = 0.10

    def reset(self) -> None:
        self._recent_evidence_history.clear()
        self._evidence_center = 0.0
        self._mb_center = 0.0

    def decide(self, state: AgentInternalState, minimum_confidence: float | None = None,
               metabolic_modifier: float = 0.0) -> DecoupledDecision:
        conf_thresh = minimum_confidence if minimum_confidence is not None else self.base_confidence
        lptc_ev = state.vs_net
        retina_ev = state.retina_velocity + 0.30 * state.retina_acceleration
        cx_ev = state.cx_heading
        mb_ev = state.mb_valence
        mb_wait_q, mb_up_q, mb_down_q = state.mb_action_values
        streams = (lptc_ev, retina_ev, cx_ev, mb_ev)
        sensory_agree = lptc_ev * retina_ev > 0.0 and min(abs(lptc_ev), abs(retina_ev)) > 0.10
        conflicts: list[float] = []
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                left, right = streams[i], streams[j]
                if (left > 0.08 and right < -0.08) or (left < -0.08 and right > 0.08):
                    diff = abs(left - right)
                    if sensory_agree and (i == 3 or j == 3):
                        diff *= 0.40
                    conflicts.append(diff)
        conflict_val = sum(conflicts) / len(conflicts) if conflicts else 0.0

        p_h_up, p_h_down, p_neutral = state.hypothesis_probs
        directional_bias = p_h_up - p_h_down
        self._recent_evidence_history.append(directional_bias)
        if len(self._recent_evidence_history) > self.history_window:
            self._recent_evidence_history.pop(0)
        if len(self._recent_evidence_history) >= 2 and directional_bias != 0.0:
            sign = 1.0 if directional_bias > 0 else -1.0
            temporal_consistency = sum(1 for value in self._recent_evidence_history if value * sign > 0) / len(self._recent_evidence_history)
        else:
            temporal_consistency = 1.0 if directional_bias != 0.0 else 0.5

        combined = (
            0.30 * lptc_ev
            + 0.20 * retina_ev
            + 0.15 * cx_ev
            + 0.15 * mb_ev
            + 0.10 * state.fast_memory
            + 0.10 * state.slow_memory
        )
        centered_signal = combined - self._evidence_center
        centered_mb = (mb_up_q - mb_down_q) - self._mb_center
        directional = 0.70 * centered_signal + 0.30 * centered_mb
        up_score = max(0.0, directional)
        down_score = max(0.0, -directional)
        # WAIT is an abstention signal, not a permanently dominant action.
        # Cap it so one learned readout cannot erase all future coverage.
        learned_wait = min(self.wait_value_cap, max(0.0, mb_wait_q))
        confidence = max(up_score, down_score)
        self._evidence_center = (1.0 - self.center_learning_rate) * self._evidence_center + self.center_learning_rate * combined
        self._mb_center = (1.0 - self.center_learning_rate) * self._mb_center + self.center_learning_rate * (mb_up_q - mb_down_q)

        threshold = conf_thresh
        if state.coherence < 0.25:
            threshold += (0.25 - state.coherence) * 0.15
        threshold += max(0.0, state.arousal - 0.20) * 0.12 + conflict_val * 0.18 + metabolic_modifier
        threshold += 0.10 * learned_wait
        if state.regime == "RANGE":
            threshold += 0.04
        if temporal_consistency >= 0.80 and confidence > 0.10:
            threshold = max(0.08, threshold - 0.03)
        p_wait, p_up, p_down = _action_probabilities(up_score, down_score, learned_wait)

        if state.is_shock or state.regime == "SHOCK":
            return self._wait(DecisionReason.WAIT_REGIME_SHOCK, up_score, down_score, confidence, conflict_val, state, temporal_consistency, p_wait, p_up, p_down)
        if p_neutral > 0.60 and confidence < 0.12:
            return self._wait(DecisionReason.WAIT_NEUTRAL_REGIME, up_score, down_score, confidence, conflict_val, state, temporal_consistency, p_wait, p_up, p_down)
        if conflict_val >= self.max_conflict_tolerance:
            return self._wait(DecisionReason.WAIT_HIGH_CONFLICT, up_score, down_score, confidence, conflict_val, state, temporal_consistency, p_wait, p_up, p_down)
        if state.uncertainty >= self.max_uncertainty_tolerance:
            return self._wait(DecisionReason.WAIT_HIGH_UNCERTAINTY, up_score, down_score, confidence, conflict_val, state, temporal_consistency, p_wait, p_up, p_down)
        if confidence < threshold or learned_wait > max(up_score, down_score) + self.wait_decision_margin:
            return self._wait(DecisionReason.WAIT_LOW_CONFIDENCE, up_score, down_score, confidence, conflict_val, state, temporal_consistency, p_wait, p_up, p_down)

        action = Prediction.UP if up_score > down_score else Prediction.DOWN
        reason = DecisionReason.COMMIT_UP if action == Prediction.UP else DecisionReason.COMMIT_DOWN
        return DecoupledDecision(action, reason, up_score, down_score, confidence, False, conflict_val, state.uncertainty, temporal_consistency, p_wait, p_up, p_down, state.regime)

    @staticmethod
    def _wait(reason: DecisionReason, up: float, down: float, confidence: float, conflict: float,
              state: AgentInternalState, consistency: float, p_wait: float, p_up: float,
              p_down: float) -> DecoupledDecision:
        return DecoupledDecision(Prediction.WAIT, reason, up, down, confidence, True, conflict,
                                 state.uncertainty, consistency, p_wait, p_up, p_down, state.regime)


def _action_probabilities(up_score: float, down_score: float, wait_score: float) -> tuple[float, float, float]:
    # Conservative probabilities: raw neural scores are not probabilities.
    # Keep directional confidence near 0.5 until a large, stable margin exists.
    margin = up_score - down_score
    p_direction_up = 0.5 + 0.25 * math.tanh(margin / 0.60)
    wait_mass = min(0.45, max(0.05, wait_score * 0.50))
    directional_mass = 1.0 - wait_mass
    return (wait_mass, directional_mass * p_direction_up, directional_mass * (1.0 - p_direction_up))
