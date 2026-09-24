"""Sparse Mushroom Body memory with causal action-value plasticity."""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class MushroomBodyOutput:
    valence: float
    active_kc_indices: tuple[int, ...]
    novelty: float
    memory_load: float
    action_values: tuple[float, float, float] = (0.0, 0.0, 0.0)


class MushroomBodyAssociativeMemory:
    """Sparse KC memory with separate WAIT, UP and DOWN MBON readouts."""

    def __init__(
        self,
        input_dim: int = 8,
        kc_count: int = 256,
        sparsity_fraction: float = 0.05,
        learning_rate: float = 0.08,
        weight_decay: float = 0.002,
        homeostatic_target_norm: float = 3.0,
        seed: int = 42,
        enabled: bool = True,
    ) -> None:
        if kc_count < 16:
            raise ValueError("kc_count must be at least 16")
        if not 0.0 < sparsity_fraction <= 0.5:
            raise ValueError("sparsity_fraction must be in (0, 0.5]")
        self.input_dim = input_dim
        self.kc_count = kc_count
        self.k_active = max(1, int(round(kc_count * sparsity_fraction)))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.homeostatic_target_norm = homeostatic_target_norm
        self.enabled = enabled

        rng = np.random.RandomState(seed) if np is not None else None
        if np is not None:
            raw_weights = rng.randn(input_dim, kc_count).astype(np.float32)
            norms = np.linalg.norm(raw_weights, axis=0, keepdims=True)
            self._projection_np = raw_weights / np.maximum(1e-6, norms)
            self._action_weights_np = np.zeros((3, kc_count), dtype=np.float32)
            self._mbon_weights_np = np.zeros(kc_count, dtype=np.float32)
            self._kc_history_np = np.zeros(kc_count, dtype=np.float32)
        else:
            self._projection_np = None
            self._action_weights_list = [[0.0] * kc_count for _ in range(3)]
            self._mbon_weights_np = [0.0] * kc_count
            self._kc_history_list = [0.0] * kc_count
        self._pending_active_kcs: tuple[int, ...] | None = None

    def reset(self, preserve_weights: bool = False) -> None:
        if not preserve_weights:
            if np is not None and hasattr(self, "_action_weights_np"):
                self._action_weights_np.fill(0.0)
                self._mbon_weights_np.fill(0.0)
                self._kc_history_np.fill(0.0)
            else:
                self._action_weights_list = [[0.0] * self.kc_count for _ in range(3)]
                self._mbon_weights_np = [0.0] * self.kc_count
                self._kc_history_list = [0.0] * self.kc_count
        elif np is not None and hasattr(self, "_kc_history_np"):
            self._kc_history_np.fill(0.0)
        else:
            self._kc_history_list = [0.0] * self.kc_count
        self._pending_active_kcs = None

    def perceive(self, context_vector: tuple[float, ...] | np.ndarray) -> MushroomBodyOutput:
        if not self.enabled:
            return MushroomBodyOutput(0.0, (), 0.0, 0.0)
        if np is None:
            return MushroomBodyOutput(0.0, (), 0.0, 0.0)

        ctx = np.array(context_vector, dtype=np.float32)
        if len(ctx) != self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: min(len(ctx), self.input_dim)] = ctx[: self.input_dim]
            ctx = padded
        potentials = np.dot(ctx, self._projection_np)
        top_k = np.argpartition(potentials, -self.k_active)[-self.k_active:]
        active = tuple(int(idx) for idx in top_k)
        q_values = tuple(
            float(np.sum(self._action_weights_np[action, top_k]))
            for action in range(3)
        )
        action_values = tuple(math.tanh(value) for value in q_values)
        valence = math.tanh(q_values[1] - q_values[2])
        novelty = float(np.sum(self._kc_history_np[top_k] < 2.0)) / self.k_active
        self._kc_history_np[top_k] += 1.0
        self._pending_active_kcs = active
        memory_load = float(np.mean(np.abs(self._action_weights_np)))
        return MushroomBodyOutput(valence, active, novelty, memory_load, action_values)

    def reinforce_actions(self, rewards: tuple[float, float, float]) -> None:
        """Update all action readouts from counterfactual next-candle rewards."""
        if not self.enabled or self._pending_active_kcs is None:
            return
        if len(rewards) != 3:
            raise ValueError("rewards must contain WAIT, UP and DOWN")
        active = list(self._pending_active_kcs)
        if np is not None and hasattr(self, "_action_weights_np"):
            for action, reward in enumerate(rewards):
                row = self._action_weights_np[action]
                row[active] *= 1.0 - self.weight_decay
                row[active] += self.learning_rate * max(-1.0, min(1.0, reward))
            norm = float(np.linalg.norm(self._action_weights_np))
            if norm > self.homeostatic_target_norm:
                self._action_weights_np *= self.homeostatic_target_norm / norm
            else:
                np.clip(self._action_weights_np, -2.0, 2.0, out=self._action_weights_np)
            self._mbon_weights_np[:] = self._action_weights_np[1] - self._action_weights_np[2]
        else:
            for action, reward in enumerate(rewards):
                for idx in active:
                    weight = self._action_weights_list[action][idx] * (1.0 - self.weight_decay)
                    self._action_weights_list[action][idx] = max(-2.0, min(2.0, weight + self.learning_rate * reward))
            self._mbon_weights_np = [
                up - down
                for up, down in zip(self._action_weights_list[1], self._action_weights_list[2])
            ]
        self._pending_active_kcs = None

    def reinforce_action(self, action: int, reward: float) -> None:
        """Compatibility helper for a single selected-action update."""
        rewards = [0.0, 0.0, 0.0]
        rewards[action] = reward
        self.reinforce_actions(tuple(rewards))

    def reinforce(self, observed_return: float) -> None:
        """Legacy directional reinforcement mapped to UP/DOWN counterfactuals."""
        signal = math.tanh(observed_return * 20.0)
        self.reinforce_actions((0.0, signal, -signal))
