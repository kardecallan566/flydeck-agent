"""Test homeostatic synaptic scaling in Mushroom Body."""
import math
import numpy as np
import pytest

from flydeck.mushroom_body import MushroomBodyAssociativeMemory


def test_homeostatic_scaling_bounds_weights_over_extended_session() -> None:
    target_norm = 2.5
    mb = MushroomBodyAssociativeMemory(
        input_dim=8,
        kc_count=128,
        learning_rate=0.15,
        homeostatic_target_norm=target_norm,
    )

    # Simulate 500 consecutive rounds of high-reward signals
    for i in range(500):
        # Varying context
        ctx = tuple(math.sin(i * 0.1 + j) for j in range(8))
        mb.perceive(ctx)
        # Strong directional return
        ret = 0.5 if (i % 2 == 0) else -0.4
        mb.reinforce(ret)

    norm = float(np.linalg.norm(mb._mbon_weights_np))
    assert norm <= target_norm + 1e-4, f"Norm {norm} exceeded homeostatic bound {target_norm}"
    # Verify weights didn't collapse to zero either
    assert norm > 0.5, f"Norm {norm} should maintain active learned contrast"
