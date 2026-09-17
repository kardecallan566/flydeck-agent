"""Explicit and compact continuous internal state for the FlyDeck Agent.

Maintains perceptual state, motion dynamics, Central Complex context,
associative memory trace, continuous expectation, prediction error,
arousal, confidence, and uncertainty across consecutive 5-minute rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(slots=True)
class AgentInternalState:
    """Continuous internal neural state vector flowing causally across rounds."""
    
    # 1. Perceptual & Sensory Kinematics
    perceptual_balance: float = 0.0     # ON/OFF firing balance in [-1, +1]
    retina_velocity: float = 0.0        # Filtered return velocity in [-1, +1]
    retina_acceleration: float = 0.0    # Second-order momentum in [-1, +1]
    volume_contrast: float = 1.0        # Arousal/gain factor from volume surge
    volatility_contrast: float = 1.0    # Local price dispersion
    coherence: float = 0.5              # Spatial-temporal motion coherence in [0, 1]

    # 2. Motion Flow (LPTC)
    vs_net: float = 0.0                 # Vertical system opponent motion in [-1, +1]
    hs_net: float = 0.0                 # Horizontal system lateral motion in [-1, +1]
    motion_energy: float = 0.0          # Wide-field optical flow energy

    # 3. Central Complex Context
    cx_fast_bias: float = 0.0           # Fast tactical momentum (~15-30m)
    cx_slow_bias: float = 0.0           # Slow structural trend (~2-4h)
    cx_heading: float = 0.0             # Persistent ring attractor heading in [-1, +1]

    # 4. Associative Memory (Mushroom Body)
    mb_valence: float = 0.0             # Learned associative valence from MBONs in [-1, +1]
    mb_novelty: float = 0.0             # Familiarity/novelty of current sparse KC representation

    # 5. Predictive Coding & Expectation
    expectation: float = 0.0            # Continuous expected market dynamic in [-1, +1]
    prediction_error: float = 0.0       # Surprise magnitude |observed - expectation| in [0, 2]
    signed_error: float = 0.0           # Signed error (observed - expectation) in [-2, +2]

    # 6. Neuromodulation & Internal Drive
    arousal: float = 0.20               # Octopaminergic arousal (volatility + surprise)
    dopamine_burst: float = 0.0         # Transient novelty / expectation violation signal

    # 7. Uncertainty & Metacognition
    uncertainty: float = 0.0            # Dispersion across competing hypotheses [0, 1]
    conflict: float = 0.0               # Pairwise disagreement across active pathways [0, 1]
    temporal_consistency: float = 0.0   # Stability of hypothesis over sliding window

    # Hypothesis distribution (UP, DOWN, CHOP)
    hypothesis_probs: tuple[float, float, float] = (0.333, 0.333, 0.334)

    def copy(self) -> AgentInternalState:
        return AgentInternalState(
            perceptual_balance=self.perceptual_balance,
            retina_velocity=self.retina_velocity,
            retina_acceleration=self.retina_acceleration,
            volume_contrast=self.volume_contrast,
            volatility_contrast=self.volatility_contrast,
            coherence=self.coherence,
            vs_net=self.vs_net,
            hs_net=self.hs_net,
            motion_energy=self.motion_energy,
            cx_fast_bias=self.cx_fast_bias,
            cx_slow_bias=self.cx_slow_bias,
            cx_heading=self.cx_heading,
            mb_valence=self.mb_valence,
            mb_novelty=self.mb_novelty,
            expectation=self.expectation,
            prediction_error=self.prediction_error,
            signed_error=self.signed_error,
            arousal=self.arousal,
            dopamine_burst=self.dopamine_burst,
            uncertainty=self.uncertainty,
            conflict=self.conflict,
            temporal_consistency=self.temporal_consistency,
            hypothesis_probs=self.hypothesis_probs,
        )

    def to_diagnostic_dict(self) -> dict[str, float]:
        return {
            "perceptual_balance": self.perceptual_balance,
            "retina_velocity": self.retina_velocity,
            "vs_net": self.vs_net,
            "cx_heading": self.cx_heading,
            "mb_valence": self.mb_valence,
            "expectation": self.expectation,
            "prediction_error": self.prediction_error,
            "arousal": self.arousal,
            "conflict": self.conflict,
            "uncertainty": self.uncertainty,
            "p_up": self.hypothesis_probs[0],
            "p_down": self.hypothesis_probs[1],
            "p_chop": self.hypothesis_probs[2],
        }
