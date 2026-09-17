"""Neural diagnostics and explainability tracer for FlyDeck Agent.

Maintains an auditable step-by-step history of:
- Internal state dynamics (motion, CX heading, expectation, arousal).
- Active Kenyon Cells (Mushroom Body sparse representation).
- Cross-circuit consensus vs conflict breakdown.
- Decision reasons (UP, DOWN, or specific WAIT rationale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .decision_engine import DecoupledDecision, DecisionReason
from .internal_state import AgentInternalState


@dataclass(frozen=True, slots=True)
class CandleDiagnosticEntry:
    round_index: int
    close_price: float
    action: str
    reason: str
    confidence: float
    conflict: float
    uncertainty: float
    temporal_consistency: float
    prediction_error: float
    arousal: float
    cx_heading: float
    vs_net: float
    mb_valence: float
    expectation: float
    p_up: float
    p_down: float
    p_neutral: float


class NeuralDiagnosticsTracer:
    """Collects and formats diagnostic neural state traces across execution."""

    def __init__(self, max_history: int = 2000) -> None:
        self.max_history = max_history
        self._entries: list[CandleDiagnosticEntry] = []

    def reset(self) -> None:
        self._entries.clear()

    def record(
        self,
        round_index: int,
        close_price: float,
        state: AgentInternalState,
        decision: DecoupledDecision,
    ) -> CandleDiagnosticEntry:
        entry = CandleDiagnosticEntry(
            round_index=round_index,
            close_price=close_price,
            action=decision.action.name,
            reason=decision.reason.value,
            confidence=decision.confidence,
            conflict=decision.conflict,
            uncertainty=decision.uncertainty,
            temporal_consistency=decision.temporal_consistency,
            prediction_error=state.prediction_error,
            arousal=state.arousal,
            cx_heading=state.cx_heading,
            vs_net=state.vs_net,
            mb_valence=state.mb_valence,
            expectation=state.expectation,
            p_up=state.hypothesis_probs[0],
            p_down=state.hypothesis_probs[1],
            p_neutral=state.hypothesis_probs[2],
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_history:
            self._entries.pop(0)
        return entry

    @property
    def entries(self) -> tuple[CandleDiagnosticEntry, ...]:
        return tuple(self._entries)

    def summary_table(self, last_n: int = 15) -> str:
        """Format the most recent diagnostic steps as a table."""
        sample = self._entries[-last_n:] if last_n <= len(self._entries) else self._entries
        lines = [
            f"{'Rnd':<5} | {'Action':<6} | {'Reason':<22} | {'Conf':<6} | {'Err':<6} | {'Arousal':<7} | {'CX_Head':<7} | {'MB_Val':<7} | {'Conflict':<8}",
            "-" * 95,
        ]
        for e in sample:
            lines.append(
                f"{e.round_index:<5} | {e.action:<6} | {e.reason:<22} | {e.confidence:<6.2f} | "
                f"{e.prediction_error:<6.2f} | {e.arousal:<7.2f} | {e.cx_heading:<7.2f} | "
                f"{e.mb_valence:<7.2f} | {e.conflict:<8.2f}"
            )
        return "\n".join(lines)
