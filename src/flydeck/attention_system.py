"""Top-down selective visual attention system inspired by Drosophila.

In Drosophila:
- Neuromodulatory projections (dopaminergic & serotonergic) from the central complex
  and lateral horn modulate the gain of lamina (L1/L2) and medulla columns.
- Under high arousal or unexpected motion bursts, attention narrows temporally to the most
  recent sensory inputs while suppressing background spatial noise.
- Under calm / low volatility regimes, attention broadens to capture structural patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AttentionState:
    temporal_focus: float      # Weight given to immediate recent candles vs history in [0.5, 2.0]
    sensory_gain: float        # Multiplier on entry drive / excitability in [0.7, 1.5]
    suppression_factor: float  # Attenuation of minor spatial noise in [0.0, 0.5]


class TopDownAttentionModule:
    """Top-down selective attention modulating sensory processing."""

    def __init__(
        self,
        base_gain: float = 1.0,
        arousal_sensitivity: float = 0.50,
        enabled: bool = True,
    ) -> None:
        self.base_gain = base_gain
        self.arousal_sensitivity = arousal_sensitivity
        self.enabled = enabled

        self._temporal_focus = 1.0
        self._sensory_gain = base_gain
        self._suppression = 0.10

    def reset(self) -> None:
        self._temporal_focus = 1.0
        self._sensory_gain = self.base_gain
        self._suppression = 0.10

    def step(
        self,
        arousal: float,
        prediction_error: float,
        uncertainty: float,
        volatility: float,
    ) -> AttentionState:
        """Compute top-down attention parameters based on internal state."""
        if not self.enabled:
            return AttentionState(temporal_focus=1.0, sensory_gain=1.0, suppression_factor=0.0)

        # High arousal + surprise tightens temporal focus (focus on what is happening NOW)
        drive = max(0.0, arousal - 0.20) + 0.50 * prediction_error
        target_focus = 1.0 + min(1.0, drive * self.arousal_sensitivity)

        # High uncertainty suppresses background noise to avoid acting on hallucinations
        target_suppression = min(0.40, uncertainty * 0.35 + (0.10 if volatility > 0.005 else 0.0))

        # Sensory gain increases with clear directional momentum, decreases in chaotic chop
        gain_mod = (1.0 - 0.30 * uncertainty) * (1.0 + 0.30 * min(1.0, arousal))
        target_gain = max(0.70, min(1.40, self.base_gain * gain_mod))

        # Smooth temporal transition
        alpha = 0.35
        self._temporal_focus = (1.0 - alpha) * self._temporal_focus + alpha * target_focus
        self._sensory_gain = (1.0 - alpha) * self._sensory_gain + alpha * target_gain
        self._suppression = (1.0 - alpha) * self._suppression + alpha * target_suppression

        return AttentionState(
            temporal_focus=self._temporal_focus,
            sensory_gain=self._sensory_gain,
            suppression_factor=self._suppression,
        )
