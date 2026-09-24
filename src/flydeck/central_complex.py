"""Central Complex (CX) working memory, multi-scale integration, and neuromodulation.

Inspired by Drosophila Protocerebral Bridge (PB), Ellipsoid Body (EB) ring attractors,
and Fan-shaped Body (FB) action selection:
1. Recurrent Ring Attractor: Persistent internal heading state across rounds.
2. Multi-Scale Integrator: Fast timescale (15-30m) vs Slow timescale (2-4h).
3. Neuromodulatory Arousal (Octopaminergic): Volatility/chop tracking to adapt decision thresholds.
4. Changepoint / Novelty Detection (Dopaminergic): Transient reset on structural market shifts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class CentralComplexState:
    fast_bias: float           # Fast temporal momentum in [-1, +1]
    slow_bias: float           # Slow structural trend in [-1, +1]
    attractor_heading: float   # Ring attractor persistent heading in [-1, +1]
    arousal: float             # Neuromodulatory arousal state in [0, 1]
    novelty_signal: float      # Dopaminergic novelty/surprise burst
    alignment: float           # Concordance between fast and slow scales in [-1, +1]


class CentralComplexSystem:
    """Central Complex working memory and neuromodulation engine."""

    def __init__(
        self,
        tau_fast: float = 0.25,     # Alpha for fast momentum (~4 steps = 20m)
        tau_slow: float = 0.04,     # Alpha for slow trend (~25 steps = ~2h)
        attractor_leak: float = 0.15, # Decay rate for ring attractor in absence of input
        arousal_decay: float = 0.10,  # Decay rate for octopaminergic arousal
        enabled: bool = True,
    ) -> None:
        if not 0.0 < tau_fast <= 1.0:
            raise ValueError("tau_fast must be in (0, 1]")
        if not 0.0 < tau_slow <= 1.0:
            raise ValueError("tau_slow must be in (0, 1]")
        if tau_slow > tau_fast:
            raise ValueError("tau_slow must be <= tau_fast")

        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.attractor_leak = attractor_leak
        self.arousal_decay = arousal_decay
        self.enabled = enabled

        # Persistent internal state
        self._fast_bias = 0.0
        self._slow_bias = 0.0
        self._attractor_heading = 0.0
        self._arousal = 0.20
        self._prev_signal = 0.0

    def reset(self) -> None:
        self._fast_bias = 0.0
        self._slow_bias = 0.0
        self._attractor_heading = 0.0
        self._arousal = 0.20
        self._prev_signal = 0.0

    def step(
        self,
        sensory_signal: float,
        volatility: float = 0.003,
        coherence: float = 0.5,
        mb_feedback: float = 0.0,
        reset_heading: bool = False,
    ) -> CentralComplexState:
        """Update internal state with current sensory input, MB feedback and market conditions."""
        if reset_heading:
            self._attractor_heading = 0.0
            self._fast_bias = sensory_signal
            self._slow_bias = 0.50 * sensory_signal
            self._arousal = min(1.0, self._arousal + 0.40)

        if not self.enabled:
            return CentralComplexState(
                fast_bias=sensory_signal,
                slow_bias=sensory_signal,
                attractor_heading=sensory_signal,
                arousal=0.20,
                novelty_signal=0.0,
                alignment=1.0 if sensory_signal != 0.0 else 0.0,
            )

        # 1. Dopaminergic Novelty / Surprise: Sharp divergence between expectation and sensory input
        expectation = self._attractor_heading
        surprise = abs(sensory_signal - expectation)
        novelty = max(0.0, surprise - 0.40)  # Active burst when surprise > threshold

        # 2. Octopaminergic Arousal: Integrates volatility, surprise, and inverse coherence
        # High volatility / low coherence increases arousal -> raises threshold / caution
        arousal_input = (volatility * 100.0) * 0.40 + (1.0 - coherence) * 0.40 + novelty * 0.20
        self._arousal = (1.0 - self.arousal_decay) * self._arousal + self.arousal_decay * arousal_input
        self._arousal = max(0.05, min(1.0, self._arousal))

        # 3. Multi-Scale Integrators (Fast vs Slow)
        # In presence of novelty burst, accelerate fast adaptation to avoid fighting new trend
        effective_fast_alpha = min(0.80, self.tau_fast * (1.0 + 2.0 * novelty))
        self._fast_bias = (1.0 - effective_fast_alpha) * self._fast_bias + effective_fast_alpha * sensory_signal
        self._fast_bias = max(-1.0, min(1.0, self._fast_bias))

        self._slow_bias = (1.0 - self.tau_slow) * self._slow_bias + self.tau_slow * sensory_signal
        self._slow_bias = max(-1.0, min(1.0, self._slow_bias))

        # 4. Ring Attractor Heading (Persistent Working Memory + MB Feedback)
        base_heading = 0.65 * self._fast_bias + 0.35 * self._slow_bias
        if mb_feedback != 0.0:
            target_heading = 0.90 * base_heading + 0.10 * mb_feedback
        else:
            target_heading = base_heading
        self._attractor_heading = (1.0 - self.attractor_leak) * self._attractor_heading + self.attractor_leak * target_heading
        self._attractor_heading = max(-1.0, min(1.0, self._attractor_heading))

        # 5. Multi-Scale Concordance / Alignment
        # If fast and slow agree in sign: alignment > 0
        # If they conflict (e.g. fast bounce in downtrend): alignment < 0
        denom = max(1e-6, abs(self._fast_bias) * abs(self._slow_bias))
        alignment = (self._fast_bias * self._slow_bias) / denom
        # Scale alignment by minimum magnitude
        scale = min(abs(self._fast_bias), abs(self._slow_bias))
        scaled_alignment = alignment * scale

        self._prev_signal = sensory_signal

        return CentralComplexState(
            fast_bias=self._fast_bias,
            slow_bias=self._slow_bias,
            attractor_heading=self._attractor_heading,
            arousal=self._arousal,
            novelty_signal=novelty,
            alignment=max(-1.0, min(1.0, scaled_alignment)),
        )
