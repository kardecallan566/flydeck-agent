from __future__ import annotations

import math


class SparseMarketEncoder:
    """Cheap k-winners-take-all encoder for market observations.

    The encoder turns the dense 12-value market state into a sparse population
    code. Each feature has positive and negative channels, but only the k
    strongest feature responses survive. The previous action is carried as a
    tiny recurrent context signal, giving the trading circuit access to recent
    behavior without an LSTM, replay buffer, or large state vector.
    """

    def __init__(self, feature_count: int = 12, winners: int = 4) -> None:
        if feature_count < 1:
            raise ValueError("feature_count must be >= 1")
        if not 1 <= winners <= feature_count:
            raise ValueError("winners must be between 1 and feature_count")
        self.feature_count = feature_count
        self.winners = winners
        self.output_size = feature_count * 2 + 3
        self.reset()

    def reset(self) -> None:
        self._previous_action = 0

    def encode(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        if len(observation) != self.feature_count:
            raise ValueError("observation size does not match encoder feature count")

        responses = [max(-1.0, min(1.0, float(value))) for value in observation]
        ranked = sorted(range(self.feature_count), key=lambda index: abs(responses[index]), reverse=True)
        active = set(ranked[: self.winners])

        encoded = [0.0] * (self.feature_count * 2)
        for index in active:
            value = responses[index]
            encoded[index * 2] = max(0.0, value)
            encoded[index * 2 + 1] = max(0.0, -value)

        # Compact previous-action context: one active channel out of three.
        encoded.extend(1.0 if action == self._previous_action else 0.0 for action in range(3))
        return tuple(encoded)

    def observe_action(self, action: int) -> None:
        if not 0 <= action < 3:
            raise ValueError("action must be HOLD=0, BUY=1 or SELL=2")
        self._previous_action = action

    @property
    def active_units(self) -> int:
        return self.winners + 1

    @property
    def sparsity(self) -> float:
        return self.active_units / self.output_size
