from __future__ import annotations

from dataclasses import dataclass

from .bnb_prediction import Prediction


@dataclass(frozen=True, slots=True)
class RiskState:
    consecutive_losses: int = 0
    cooldown: int = 0
    drawdown: float = 0.0
    novelty: float = 0.0
    risk_modifier: float = 0.0


class LightweightRiskPolicy:
    """Deterministic policy overlay; it does not modify MaleCNS synapses."""

    def __init__(self, max_consecutive_losses: int = 6, cooldown_rounds: int = 2,
                 max_drawdown: float = 0.12) -> None:
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_rounds = cooldown_rounds
        self.max_drawdown = max_drawdown
        self._state = RiskState()

    @property
    def state(self) -> RiskState:
        return self._state

    def reset(self) -> None:
        self._state = RiskState()

    def before_action(self, action: Prediction, novelty: float = 0.0) -> Prediction:
        novelty = max(0.0, min(1.0, novelty))
        self._state = RiskState(self._state.consecutive_losses, max(0, self._state.cooldown - 1), self._state.drawdown, novelty, self._modifier(novelty))
        if action != Prediction.WAIT and (self._state.cooldown > 0 or self._state.drawdown >= self.max_drawdown):
            return Prediction.WAIT
        # Novelty is a soft risk modifier only. It must never be an absolute
        # entry veto; otherwise normal regime transitions collapse to WAIT.
        return action

    def observe(self, action: Prediction, reward: float) -> None:
        if action == Prediction.WAIT:
            return
        if reward < 0.0:
            losses = self._state.consecutive_losses + 1
            cooldown = self.cooldown_rounds if losses >= self.max_consecutive_losses else self._state.cooldown
            drawdown = min(1.0, self._state.drawdown + min(0.02, abs(reward) * 0.01))
        else:
            losses = 0
            cooldown = self._state.cooldown
            drawdown = max(0.0, self._state.drawdown - min(0.01, reward * 0.005))
        self._state = RiskState(losses, cooldown, drawdown, self._state.novelty, self._modifier(self._state.novelty))

    @staticmethod
    def _modifier(novelty: float) -> float:
        return min(0.12, 0.04 * novelty)
