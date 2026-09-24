"""Predictive coding loop and continuous hypothesis competition for FlyDeck Agent.

Inspired by Drosophila predictive visual and motor processing:
1. Continuous Expectation: Internal expectation of next directional change E_hat in [-1, +1].
2. Prediction Error / Surprise: delta_t = |Observed_t - Expectation_{t-1}|.
3. Competing Hypotheses: Distribution over directional hypotheses (P(UP), P(DOWN), P(NEUTRAL)).
4. Neuromodulatory Surprise: Surprise drives Dopaminergic Neurons (DANs) to recalibrate hypothesis
   probabilities and reset persistent biases upon regime change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class PredictiveCodingUpdate:
    expectation: float              # Expected sensory dynamic for the NEXT round [-1, +1]
    prediction_error: float         # Magnitude of surprise |observed - expectation| [0, 2]
    signed_error: float             # Signed prediction error (observed - expectation) [-2, +2]
    hypothesis_probs: tuple[float, float, float]  # (p_up, p_down, p_neutral)
    dopamine_burst: float           # Transient surprise burst [0, 1]
    uncertainty: float              # Shannon entropy / dispersion over hypotheses [0, 1]


class PredictiveCodingEngine:
    """True closed-loop predictive coding engine for temporal market dynamics."""

    def __init__(
        self,
        expectation_learning_rate: float = 0.25,
        surprise_threshold: float = 0.35,
        entropy_temperature: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.expectation_learning_rate = expectation_learning_rate
        self.surprise_threshold = surprise_threshold
        self.entropy_temperature = entropy_temperature
        self.enabled = enabled

        # Internal continuous expectation prior
        self._current_expectation = 0.0
        # Hypothesis logits: [logit_up, logit_down, logit_neutral]
        self._hypothesis_logits = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        self._current_expectation = 0.0
        self._hypothesis_logits = [0.0, 0.0, 0.0]

    def step(
        self,
        observed_motion: float,
        cx_context: float,
        mb_valence: float,
        volatility: float,
    ) -> PredictiveCodingUpdate:
        """Execute predictive coding cycle:
        1. Compare observed motion with prior expectation -> compute surprise.
        2. Update hypothesis competition.
        3. Formulate next expectation conditioning on current observation, CX and MB.
        """
        if not self.enabled:
            probs = (0.333, 0.333, 0.334)
            return PredictiveCodingUpdate(
                expectation=observed_motion,
                prediction_error=0.0,
                signed_error=0.0,
                hypothesis_probs=probs,
                dopamine_burst=0.0,
                uncertainty=1.0,
            )

        # 1. Error computation against expectation from prior round
        signed_error = observed_motion - self._current_expectation
        prediction_error = abs(signed_error)

        # Dopaminergic surprise burst when error exceeds threshold
        dopamine_burst = max(0.0, min(1.0, (prediction_error - self.surprise_threshold) * 2.0))

        # 2. Update hypothesis competition:
        # Evidence: observed_motion (sensory: 60%), cx_context (heading: 25%), mb_valence (associative memory: 15%)
        ev_up = max(0.0, observed_motion) * 0.60 + max(0.0, cx_context) * 0.25 + max(0.0, mb_valence) * 0.15
        ev_down = max(0.0, -observed_motion) * 0.60 + max(0.0, -cx_context) * 0.25 + max(0.0, -mb_valence) * 0.15
        # Neutral / chop evidence: low net motion or conflicting signals
        ev_neutral = max(0.0, 1.0 - (abs(observed_motion) + abs(cx_context)) * 0.5)

        # If surprise is high, leak / decay prior hypothesis logits (prevent momentum trap)
        decay = 0.65 if dopamine_burst > 0.3 else 0.85
        self._hypothesis_logits[0] = self._hypothesis_logits[0] * decay + ev_up
        self._hypothesis_logits[1] = self._hypothesis_logits[1] * decay + ev_down
        self._hypothesis_logits[2] = self._hypothesis_logits[2] * decay + ev_neutral

        # Softmax over hypothesis logits
        logits = self._hypothesis_logits
        max_l = max(logits)
        exp_vals = [math.exp((l - max_l) / max(0.1, self.entropy_temperature)) for l in logits]
        sum_exp = sum(exp_vals)
        probs = (exp_vals[0] / sum_exp, exp_vals[1] / sum_exp, exp_vals[2] / sum_exp)

        # Normalized Shannon entropy [0, 1] as uncertainty metric
        ent = -sum(p * math.log(max(1e-9, p)) for p in probs) / math.log(3.0)

        # 3. Formulate expectation for NEXT candle:
        # Integrates current sensory state with associative memory and CX heading
        next_target = (
            0.45 * observed_motion
            + 0.35 * cx_context
            + 0.20 * mb_valence
        )
        lr = self.expectation_learning_rate
        # Surprise accelerates learning rate of expectation
        eff_lr = min(0.85, lr * (1.0 + 2.0 * dopamine_burst))
        self._current_expectation = (1.0 - eff_lr) * self._current_expectation + eff_lr * next_target
        self._current_expectation = max(-1.0, min(1.0, self._current_expectation))

        return PredictiveCodingUpdate(
            expectation=self._current_expectation,
            prediction_error=prediction_error,
            signed_error=signed_error,
            hypothesis_probs=probs,
            dopamine_burst=dopamine_burst,
            uncertainty=max(0.0, min(1.0, ent)),
        )
