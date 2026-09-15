"""Lobula Plate Tangential Cells (LPTC) system for wide-field motion integration.

In Drosophila, thousands of local elementary motion detectors (T4 and T5)
converge onto a small set of wide-field Lobula Plate Tangential Cells:
- Vertical System (VS): Integrates vertical optic flow (pitch/heave -> price UP vs DOWN)
  with strong opponent dendritic/axonal inhibition between UP and DOWN subtrees.
- Horizontal System (HS): Integrates progressive vs regressive horizontal flow
  (persistence vs reversal).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class LPTCOutput:
    vs_up: float               # Opponent vertical UP response
    vs_down: float             # Opponent vertical DOWN response
    vs_net: float              # Signed net vertical flow in [-1, +1]
    hs_net: float              # Signed net horizontal flow in [-1, +1]
    motion_energy: float       # Total wide-field motion energy
    directional_coherence: float  # Asymmetry ratio (purity of motion)


class LobulaPlateTangentialSystem:
    """Wide-field pooling and opponent integration of T4/T5 motion detectors."""

    def __init__(
        self,
        opponent_inhibition: float = 0.45,
        temporal_smoothing: float = 0.60,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= opponent_inhibition <= 1.0:
            raise ValueError("opponent_inhibition must be in [0, 1]")
        if not 0.0 < temporal_smoothing <= 1.0:
            raise ValueError("temporal_smoothing must be in (0, 1]")

        self.opponent_inhibition = opponent_inhibition
        self.temporal_smoothing = temporal_smoothing
        self.enabled = enabled

        self._smooth_vs_up = 0.0
        self._smooth_vs_down = 0.0
        self._smooth_hs_net = 0.0

    def reset(self) -> None:
        self._smooth_vs_up = 0.0
        self._smooth_vs_down = 0.0
        self._smooth_hs_net = 0.0

    def step(
        self,
        t4_activity: tuple[float, float, float, float],
        t5_activity: tuple[float, float, float, float],
    ) -> LPTCOutput:
        """Pool T4 and T5 directional activities and compute wide-field LPTC outputs.

        Indices in t4/t5 activities:
        0: 'a' (right / persistence)
        1: 'b' (left / reversal)
        2: 'c' (upward motion)
        3: 'd' (downward motion)
        """
        if not self.enabled:
            # Fallback un-pooled pass
            raw_up = max(0.0, t4_activity[2] + t5_activity[2])
            raw_down = max(0.0, t4_activity[3] + t5_activity[3])
            tot = raw_up + raw_down
            net = (raw_up - raw_down) / tot if tot > 1e-12 else 0.0
            return LPTCOutput(
                vs_up=raw_up,
                vs_down=raw_down,
                vs_net=net,
                hs_net=0.0,
                motion_energy=tot,
                directional_coherence=abs(net),
            )

        # 1. Raw Wide-Field Excitatory Pools
        # Combine ON (T4) and OFF (T5) pathways with synergetic synergy
        raw_up = (t4_activity[2] + t5_activity[2]) * 0.5
        raw_down = (t4_activity[3] + t5_activity[3]) * 0.5

        raw_right = (t4_activity[0] + t5_activity[0]) * 0.5
        raw_left = (t4_activity[1] + t5_activity[1]) * 0.5

        # 2. Opponent Inhibition in Vertical System (VS)
        gamma_opp = self.opponent_inhibition
        opp_up = max(0.0, raw_up - gamma_opp * raw_down)
        opp_down = max(0.0, raw_down - gamma_opp * raw_up)

        # 3. Horizontal System (HS) Progressive vs Regressive
        hs_raw = raw_right - raw_left

        # 4. Temporal Low-Pass Filtering (Dendritic integration time constant)
        alpha = self.temporal_smoothing
        self._smooth_vs_up = (1.0 - alpha) * self._smooth_vs_up + alpha * opp_up
        self._smooth_vs_down = (1.0 - alpha) * self._smooth_vs_down + alpha * opp_down
        self._smooth_hs_net = (1.0 - alpha) * self._smooth_hs_net + alpha * hs_raw

        tot_vs = self._smooth_vs_up + self._smooth_vs_down
        vs_net = (self._smooth_vs_up - self._smooth_vs_down) / tot_vs if tot_vs > 1e-12 else 0.0

        tot_hs = raw_right + raw_left
        hs_net = self._smooth_hs_net / tot_hs if tot_hs > 1e-12 else 0.0

        motion_energy = tot_vs + tot_hs
        coherence = abs(vs_net)

        return LPTCOutput(
            vs_up=self._smooth_vs_up,
            vs_down=self._smooth_vs_down,
            vs_net=max(-1.0, min(1.0, vs_net)),
            hs_net=max(-1.0, min(1.0, hs_net)),
            motion_energy=motion_energy,
            directional_coherence=coherence,
        )
