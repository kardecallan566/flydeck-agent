"""Metabolic risk controller inspired by Drosophila nutritional state modulation.

In Drosophila:
- Energy reserves and recent feeding success modulate foraging persistence and risk-taking.
- A well-nourished fly in a rich patch exploits consistently, while a fly facing sudden depletion
  becomes highly selective and cautious to conserve vital resources.

In FlyDeck:
- Maintains an internal metabolic energy state E in [0.5, 1.5] updated causally from realized returns.
- Adapts decision confidence threshold and tolerance dynamically:
  * High energy (recent confirmed wins): slight threshold relaxation to harvest trend continuation.
  * Low energy (recent losses / drawdown): elevates threshold to force defensive WAIT.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class MetabolicState:
    energy_level: float        # Current metabolic energy in [0.5, 1.5]
    threshold_modifier: float  # Multiplier or delta on confidence threshold [-0.04, +0.06]
    risk_appetite: float       # Risk tolerance scaling factor in [0.6, 1.4]


class MetabolicRiskController:
    """Modulates risk appetite and decision selectivity based on energetic state."""

    def __init__(
        self,
        base_energy: float = 1.0,
        decay_rate: float = 0.05,
        energy_gain: float = 0.15,
        enabled: bool = True,
    ) -> None:
        self.base_energy = base_energy
        self.decay_rate = decay_rate
        self.energy_gain = energy_gain
        self.enabled = enabled

        self._energy = base_energy

    def reset(self) -> None:
        self._energy = self.base_energy

    def update_feedback(self, realized_return: float) -> None:
        """Causally update energy level upon observing realized outcome."""
        if not self.enabled:
            return

        # Positive outcome feeds the agent; negative outcome depletes reserves
        normalized_reward = math.tanh(realized_return * 10.0)
        delta = self.energy_gain * normalized_reward
        # Gentle homeostatic decay toward base energy
        self._energy += delta - self.decay_rate * (self._energy - self.base_energy)
        self._energy = max(0.50, min(1.50, self._energy))

    def step(self) -> MetabolicState:
        """Read out metabolic parameters to modulate decision engine."""
        if not self.enabled:
            return MetabolicState(energy_level=1.0, threshold_modifier=0.0, risk_appetite=1.0)

        # High energy (>1.0) lowers threshold up to -0.03 (moderate relaxation)
        # Low energy (<1.0) raises threshold up to +0.05 (defensive protection)
        diff = self._energy - 1.0
        threshold_modifier = -0.04 * diff if diff > 0 else -0.06 * diff
        risk_appetite = max(0.60, min(1.40, self._energy))

        return MetabolicState(
            energy_level=self._energy,
            threshold_modifier=threshold_modifier,
            risk_appetite=risk_appetite,
        )
