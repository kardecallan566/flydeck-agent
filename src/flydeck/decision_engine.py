"""Decoupled decision engine evaluating temporal consistency, uncertainty, and conflict.

Separates perception from action selection:
- Perceptual streams (LPTC, Retina, CX, MB) provide distributed continuous evidences.
- Evaluates temporal hypothesis consistency over sliding windows.
- Quantifies cross-circuit conflict and information uncertainty.
- Emits UP, DOWN, or WAIT with explicit causal justification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

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


class DynamicDecisionEngine:
    """Decoupled action selection engine for FlyDeck Agent."""

    def __init__(
        self,
        base_confidence: float = 0.15,
        max_conflict_tolerance: float = 0.35,
        max_uncertainty_tolerance: float = 0.88,
        history_window: int = 5,
        enabled: bool = True,
    ) -> None:
        self.base_confidence = base_confidence
        self.max_conflict_tolerance = max_conflict_tolerance
        self.max_uncertainty_tolerance = max_uncertainty_tolerance
        self.history_window = history_window
        self.enabled = enabled

        self._recent_evidence_history: list[float] = []

    def reset(self) -> None:
        self._recent_evidence_history.clear()

    def decide(
        self,
        state: AgentInternalState,
        minimum_confidence: float | None = None,
        metabolic_modifier: float = 0.0,
    ) -> DecoupledDecision:
        """Evaluate internal state and select action: UP, DOWN, or WAIT."""
        conf_thresh = minimum_confidence if minimum_confidence is not None else self.base_confidence

        # 1. Multi-Stream Evidence Extraction
        lptc_ev = state.vs_net
        retina_ev = state.retina_velocity + 0.30 * state.retina_acceleration
        cx_ev = state.cx_heading
        mb_ev = state.mb_valence

        streams = (lptc_ev, retina_ev, cx_ev, mb_ev)

        # 2. Conflict Calculation across distinct functional systems
        conflicts: list[float] = []
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                s_i, s_j = streams[i], streams[j]
                if (s_i > 0.08 and s_j < -0.08) or (s_i < -0.08 and s_j > 0.08):
                    conflicts.append(abs(s_i - s_j))
        conflict_val = sum(conflicts) / len(conflicts) if conflicts else 0.0

        # 3. Hypothesis Probability Extraction
        p_up, p_down, p_neutral = state.hypothesis_probs
        directional_bias = p_up - p_down

        # 4. Temporal Consistency over sliding window
        self._recent_evidence_history.append(directional_bias)
        if len(self._recent_evidence_history) > self.history_window:
            self._recent_evidence_history.pop(0)

        # Sign consistency: fraction of recent steps sharing the same sign
        if len(self._recent_evidence_history) >= 2:
            current_sign = 1.0 if directional_bias > 0 else (-1.0 if directional_bias < 0 else 0.0)
            if current_sign != 0.0:
                matching = sum(1 for v in self._recent_evidence_history if (v * current_sign) > 0)
                temporal_consistency = matching / len(self._recent_evidence_history)
            else:
                temporal_consistency = 0.5
        else:
            temporal_consistency = 1.0

        # 5. Continuous Confidence & Score Formation
        # Combined evidence weighted by consensus
        combined_signal = (
            0.35 * lptc_ev
            + 0.25 * retina_ev
            + 0.20 * cx_ev
            + 0.20 * mb_ev
        )
        up_score = max(0.0, combined_signal)
        down_score = max(0.0, -combined_signal)
        confidence = abs(combined_signal)

        # Dynamic adaptive threshold modulated by:
        # - Coherence
        # - Neuromodulatory Arousal
        # - Conflict
        # - Temporal consistency
        adaptive_thresh = conf_thresh
        if state.coherence < 0.25:
            adaptive_thresh += (0.25 - state.coherence) * 0.15
        adaptive_thresh += max(0.0, state.arousal - 0.20) * 0.12
        adaptive_thresh += conflict_val * 0.18
        adaptive_thresh += metabolic_modifier

        # Bonus threshold relaxation if temporal consistency is very high
        if temporal_consistency >= 0.80 and confidence > 0.10:
            adaptive_thresh = max(0.08, adaptive_thresh - 0.03)

        # 6. Action Selection with Reason Attribution
        # Emergency escape triggered by Giant Fiber circuit on looming shock
        if state.is_shock:
            return DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_REGIME_SHOCK,
                up_score=up_score,
                down_score=down_score,
                confidence=confidence,
                wait=True,
                conflict=conflict_val,
                uncertainty=state.uncertainty,
                temporal_consistency=temporal_consistency,
            )

        if p_neutral > 0.60 and confidence < 0.12:
            return DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_NEUTRAL_REGIME,
                up_score=up_score,
                down_score=down_score,
                confidence=confidence,
                wait=True,
                conflict=conflict_val,
                uncertainty=state.uncertainty,
                temporal_consistency=temporal_consistency,
            )

        if conflict_val >= self.max_conflict_tolerance:
            return DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_HIGH_CONFLICT,
                up_score=up_score,
                down_score=down_score,
                confidence=confidence,
                wait=True,
                conflict=conflict_val,
                uncertainty=state.uncertainty,
                temporal_consistency=temporal_consistency,
            )

        if state.uncertainty >= self.max_uncertainty_tolerance:
            return DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_HIGH_UNCERTAINTY,
                up_score=up_score,
                down_score=down_score,
                confidence=confidence,
                wait=True,
                conflict=conflict_val,
                uncertainty=state.uncertainty,
                temporal_consistency=temporal_consistency,
            )

        if confidence < adaptive_thresh:
            return DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_LOW_CONFIDENCE,
                up_score=up_score,
                down_score=down_score,
                confidence=confidence,
                wait=True,
                conflict=conflict_val,
                uncertainty=state.uncertainty,
                temporal_consistency=temporal_consistency,
            )

        # Sufficient confidence & acceptable conflict: commit
        if up_score > down_score:
            action = Prediction.UP
            reason = DecisionReason.COMMIT_UP
        else:
            # Calibrate DOWN to avoid committing on noisy pullbacks
            if p_down < 0.38 and confidence < (adaptive_thresh + 0.03):
                return DecoupledDecision(
                    action=Prediction.WAIT,
                    reason=DecisionReason.WAIT_LOW_CONFIDENCE,
                    up_score=up_score,
                    down_score=down_score,
                    confidence=confidence,
                    wait=True,
                    conflict=conflict_val,
                    uncertainty=state.uncertainty,
                    temporal_consistency=temporal_consistency,
                )
            action = Prediction.DOWN
            reason = DecisionReason.COMMIT_DOWN

        return DecoupledDecision(
            action=action,
            reason=reason,
            up_score=up_score,
            down_score=down_score,
            confidence=confidence,
            wait=False,
            conflict=conflict_val,
            uncertainty=state.uncertainty,
            temporal_consistency=temporal_consistency,
        )
