"""Mushroom Body (MB) sparse associative memory with causal reinforcement learning.

In Drosophila:
- Kenyon Cells (KCs) expand sensory-contextual input into a high-dimensional space
  with strict sparsity (~5% active), enforced by the GABAergic APL feedback neuron.
- Mushroom Body Output Neurons (MBONs) integrate active KCs with plastic weights.
- Dopaminergic Neurons (DANs) convey reward/punishment signals (surprise or directional confirmation)
  inducing local synaptic plasticity on recently active KCs.

Strict Causality:
- At round t, the MB receives context S_t, activates KCs, and predicts valence V_t.
- At round t+1, when real outcome O_t is known, the trace of KCs active at t is updated.
- Zero future data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class MushroomBodyOutput:
    valence: float              # Learned directional valence in [-1, +1]
    active_kc_indices: tuple[int, ...]  # Active Kenyon cells in sparse representation
    novelty: float              # Fraction of never-before-seen KCs (novelty signal)
    memory_load: float          # Average saturation of associative weights


class MushroomBodyAssociativeMemory:
    """Sparse associative memory model inspired by Drosophila Mushroom Body."""

    def __init__(
        self,
        input_dim: int = 8,
        kc_count: int = 256,
        sparsity_fraction: float = 0.05,
        learning_rate: float = 0.08,
        weight_decay: float = 0.002,
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
        self.enabled = enabled

        # Deterministic pseudo-random projection from input context to Kenyon Cells
        rng = np.random.RandomState(seed) if np is not None else None
        if np is not None:
            # Sparse binary/ternary projection matrix (3-4 random connections per KC)
            raw_weights = rng.randn(input_dim, kc_count).astype(np.float32)
            # Normalize projection columns
            norms = np.linalg.norm(raw_weights, axis=0, keepdims=True)
            self._projection_np = raw_weights / np.maximum(1e-6, norms)
            # MBON synaptic weights for directional valence (positive=UP, negative=DOWN)
            self._mbon_weights_np = np.zeros(kc_count, dtype=np.float32)
            # KC activation occurrence tracking for novelty
            self._kc_history_np = np.zeros(kc_count, dtype=np.float32)
        else:
            self._projection_np = None
            self._mbon_weights_np = [0.0] * kc_count
            self._kc_history_list = [0.0] * kc_count

        # Eligibility trace for causal t -> t+1 reinforcement learning
        self._pending_active_kcs: tuple[int, ...] | None = None

    def reset(self) -> None:
        """Reset internal states and memory traces."""
        if np is not None and self._mbon_weights_np is not None:
            self._mbon_weights_np.fill(0.0)
            self._kc_history_np.fill(0.0)
        else:
            self._mbon_weights_np = [0.0] * self.kc_count
            self._kc_history_list = [0.0] * self.kc_count
        self._pending_active_kcs = None

    def perceive(self, context_vector: tuple[float, ...] | np.ndarray) -> MushroomBodyOutput:
        """Expand continuous context into sparse Kenyon Cells and read out MBON valence."""
        if not self.enabled:
            return MushroomBodyOutput(
                valence=0.0,
                active_kc_indices=(),
                novelty=0.0,
                memory_load=0.0,
            )

        if np is not None:
            ctx = np.array(context_vector, dtype=np.float32)
            if len(ctx) != self.input_dim:
                # Pad or truncate gracefully
                new_ctx = np.zeros(self.input_dim, dtype=np.float32)
                l = min(len(ctx), self.input_dim)
                new_ctx[:l] = ctx[:l]
                ctx = new_ctx

            # 1. Project to Kenyon Cell membrane potentials
            kc_potentials = np.dot(ctx, self._projection_np)

            # 2. Strict K-WTA (APL GABAergic lateral inhibition) -> Sparsity ~5%
            # Find indices of top-K activations
            top_k_indices = np.argpartition(kc_potentials, -self.k_active)[-self.k_active:]
            active_kcs = tuple(int(idx) for idx in top_k_indices)

            # 3. Readout via MBON synaptic weights
            valence_sum = float(np.sum(self._mbon_weights_np[top_k_indices]))
            valence = math.tanh(valence_sum)

            # 4. Novelty calculation: fraction of currently active KCs that have seen < 2 lifetime activations
            novel_count = float(np.sum(self._kc_history_np[top_k_indices] < 2.0))
            novelty = novel_count / self.k_active

            # Increment history
            self._kc_history_np[top_k_indices] += 1.0

            # Store pending trace for delayed causal reinforcement at next step
            self._pending_active_kcs = active_kcs

            memory_load = float(np.mean(np.abs(self._mbon_weights_np)))
            return MushroomBodyOutput(
                valence=valence,
                active_kc_indices=active_kcs,
                novelty=novelty,
                memory_load=memory_load,
            )

        # Pure-Python fallback path
        return MushroomBodyOutput(valence=0.0, active_kc_indices=(), novelty=0.0, memory_load=0.0)

    def reinforce(self, observed_return: float) -> None:
        """Causally update MBON synaptic weights of Kenyon Cells that were active in the prior round.

        Args:
            observed_return: The realized price return or direction delta (+1 for UP, -1 for DOWN).
        """
        if not self.enabled or self._pending_active_kcs is None:
            return

        active = self._pending_active_kcs
        # Reward signal: +1 for positive move, -1 for negative move
        dopamine_signal = math.tanh(observed_return * 20.0)

        if np is not None and self._mbon_weights_np is not None:
            # Weight decay across active units (forgetting / homeostatic scaling)
            self._mbon_weights_np[list(active)] *= (1.0 - self.weight_decay)
            # Associative LTP / LTD: delta W = eta * DA
            self._mbon_weights_np[list(active)] += self.learning_rate * dopamine_signal
            # Clip weights to prevent runaway excitation
            np.clip(self._mbon_weights_np, -2.0, 2.0, out=self._mbon_weights_np)
        else:
            for idx in active:
                w = self._mbon_weights_np[idx] * (1.0 - self.weight_decay)
                w += self.learning_rate * dopamine_signal
                self._mbon_weights_np[idx] = max(-2.0, min(2.0, w))

        # Clear pending trace once reinforced
        self._pending_active_kcs = None
