"""Giant Fiber escape circuit and regime shock detector for FlyDeck Agent.

Inspired by Drosophila Giant Fiber System:
- Mediates rapid escape reflexes when visual stimuli signal impending collision (looming shock).
- In financial markets, sudden extreme volume/price anomalies (black swans or major news spikes)
  signal that existing contextual momentum has violently collapsed.
- When triggered:
  1. Activates emergency escape (shock_active=True) for a cooldown period (e.g. 2-3 candles).
  2. Forces immediate WAIT_REGIME_SHOCK to protect capital.
  3. Resets obsolete Central Complex heading to prevent stubborn counter-trend positioning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ShockState:
    is_shock: bool             # Whether the circuit fired on the current round
    shock_magnitude: float     # Intensity of the detected anomaly in [0, 1]
    cooldown_remaining: int    # Remaining rounds of mandatory post-shock abstention


class GiantFiberEscapeCircuit:
    """Detects violent market regime shocks and enforces emergency protection."""

    def __init__(
        self,
        shock_threshold: float = 0.65,
        cooldown_rounds: int = 2,
        enabled: bool = True,
    ) -> None:
        self.shock_threshold = shock_threshold
        self.cooldown_rounds = cooldown_rounds
        self.enabled = enabled

        self._cooldown = 0
        self._last_magnitude = 0.0

    def reset(self) -> None:
        self._cooldown = 0
        self._last_magnitude = 0.0

    def step(
        self,
        current_return_pct: float,
        prediction_error: float,
        volatility_contrast: float,
        volume_contrast: float,
    ) -> ShockState:
        """Evaluate market conditions for looming regime collision."""
        if not self.enabled:
            return ShockState(is_shock=False, shock_magnitude=0.0, cooldown_remaining=0)

        # Decrement cooldown if active
        if self._cooldown > 0:
            self._cooldown -= 1

        # Shock metrics:
        # 1. Price return velocity magnitude (e.g. > 1.2% in a single 5m candle is massive in BNB)
        ret_mag = min(1.0, abs(current_return_pct) / 1.5)
        # 2. Surprise factor (prediction error)
        surprise_mag = min(1.0, prediction_error / 1.2)
        # 3. Volume surge
        vol_surge = min(1.0, max(0.0, volume_contrast - 1.0) / 2.0)

        # Combined looming threat score
        shock_magnitude = 0.45 * ret_mag + 0.35 * surprise_mag + 0.20 * vol_surge
        self._last_magnitude = shock_magnitude

        fired = False
        if shock_magnitude >= self.shock_threshold:
            fired = True
            self._cooldown = self.cooldown_rounds

        is_shock = fired or (self._cooldown > 0)
        return ShockState(
            is_shock=is_shock,
            shock_magnitude=shock_magnitude,
            cooldown_remaining=self._cooldown,
        )
