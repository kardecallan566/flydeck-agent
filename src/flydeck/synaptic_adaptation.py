"""Synaptic adaptation and resource depression for Drosophila neural dynamics.

Implements short-term synaptic plasticity (Tsodyks-Markram inspired model)
where repetitive activity depletes available neurotransmitter vesicles (habituation),
while novelty or direction reversals evoke maximal transient responses (phasic burst).
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True, slots=True)
class SynapticAdaptationConfig:
    tau_recovery: float = 8.0     # Time steps to recover 63% of depleted resources
    utilization: float = 0.35     # Fraction of resources consumed per unit of activity
    min_resource: float = 0.15    # Lower bound on available resources (prevents total silence)


class SynapticAdaptation:
    """Vectorized short-term depression (vesicle depletion) across a neuron population."""

    def __init__(
        self,
        neuron_count: int,
        config: SynapticAdaptationConfig | None = None,
        enabled: bool = True,
    ) -> None:
        if neuron_count < 1:
            raise ValueError("neuron_count must be positive")
        self.neuron_count = neuron_count
        self.config = config or SynapticAdaptationConfig()
        self.enabled = enabled

        if np is not None:
            self._resources_np = np.ones(neuron_count, dtype=np.float32)
        else:
            self._resources_np = None
        self._resources_list = [1.0] * neuron_count

    def reset(self) -> None:
        if self._resources_np is not None:
            self._resources_np.fill(1.0)
        self._resources_list = [1.0] * self.neuron_count

    def step(self, activity: list[float] | tuple[float, ...] | np.ndarray) -> tuple[float, ...]:
        """Modulate raw firing rates by synaptic availability, then update resource levels."""
        if not self.enabled:
            if isinstance(activity, (list, tuple)):
                return tuple(activity)
            return tuple(activity.tolist())

        cfg = self.config
        tau_rec = cfg.tau_recovery
        u = cfg.utilization
        min_r = cfg.min_resource

        if np is not None and isinstance(activity, np.ndarray):
            act_np = activity
            modulated = act_np * self._resources_np
            # Recovery: dR/dt = (1 - R) / tau_rec
            recovery = (1.0 - self._resources_np) / tau_rec
            # Depletion: u * act * R
            depletion = u * act_np * self._resources_np
            self._resources_np = np.clip(self._resources_np + recovery - depletion, min_r, 1.0)
            return tuple(modulated.tolist())

        # Fallback pure-Python path
        modulated_list = [0.0] * self.neuron_count
        for i in range(self.neuron_count):
            act = activity[i]
            r = self._resources_list[i]
            modulated_list[i] = act * r
            recovery = (1.0 - r) / tau_rec
            depletion = u * act * r
            self._resources_list[i] = max(min_r, min(1.0, r + recovery - depletion))
        return tuple(modulated_list)

    def modulate_np(self, activity_np: np.ndarray) -> np.ndarray:
        """In-place or fast NumPy modulation of activity vector."""
        if not self.enabled or self._resources_np is None:
            return activity_np

        cfg = self.config
        modulated = activity_np * self._resources_np
        recovery = (1.0 - self._resources_np) / cfg.tau_recovery
        depletion = cfg.utilization * activity_np * self._resources_np
        self._resources_np = np.clip(
            self._resources_np + recovery - depletion,
            cfg.min_resource,
            1.0,
        )
        return modulated

    @property
    def mean_resource(self) -> float:
        """Average vesicle availability across the population (diagnostics)."""
        if self._resources_np is not None:
            return float(np.mean(self._resources_np))
        return sum(self._resources_list) / self.neuron_count
