"""Agent checkpoint manager for FlyDeck.

Provides atomic serialization and deserialization of the complete learned state:
- Mushroom Body MBON synaptic weights and Kenyon Cell lifetime occurrence counters.
- Central Complex ring attractor heading and multi-scale momentum biases.
- Continuous Predictive Coding expectation and hypothesis distribution.
- Complete AgentInternalState vector.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from .internal_state import AgentInternalState
from .visual_agent import FlyVisualPredictionAgent


class AgentCheckpointManager:
    """Manages atomic saving and loading of learned fly brain parameters."""

    @staticmethod
    def save(
        agent: FlyVisualPredictionAgent,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        vis = agent.visual
        mb = vis.mushroom_body
        cx = vis.central_complex
        pc = vis.predictive_coding

        # Extract MBON synaptic weights and KC history
        if np is not None and hasattr(mb, "_action_weights_np"):
            action_weights = mb._action_weights_np.tolist()
            mbon_weights = mb._mbon_weights_np.tolist()
            kc_history = mb._kc_history_np.tolist()
        else:
            mbon_weights = list(mb._mbon_weights_np)
            action_weights = getattr(mb, "_action_weights_list", None)
            kc_history = list(getattr(mb, "_kc_history_list", []))

        state_dict = vis.internal_state.to_diagnostic_dict()

        payload: dict[str, Any] = {
            "version": 1,
            "metadata": metadata or {},
            "mushroom_body": {
                "mbon_weights": mbon_weights,
                "action_weights": action_weights,
                "kc_history": kc_history,
                "enabled": mb.enabled,
            },
            "central_complex": {
                "heading": cx._attractor_heading,
                "fast_bias": cx._fast_bias,
                "slow_bias": cx._slow_bias,
                "arousal": cx._arousal,
            },
            "predictive_coding": {
                "current_expectation": pc._current_expectation,
                "hypothesis_logits": pc._hypothesis_logits,
            },
            "decision_engine": {
                "recent_history": list(vis.decision_engine._recent_evidence_history),
            },
            "attention": {
                "temporal_focus": vis.attention._temporal_focus,
                "sensory_gain": vis.attention._sensory_gain,
            },
            "giant_fiber": {
                "cooldown": vis.giant_fiber._cooldown,
            },
            "metabolic_control": {
                "energy": vis.metabolic_control._energy,
            },
            "internal_state": state_dict,
        }

        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=destination.parent, delete=False, suffix=".tmp"
        ) as handle:
            json.dump(payload, handle, indent=2)
            temp_path = Path(handle.name)

        os.replace(temp_path, destination)
        return destination

    @staticmethod
    def load(
        agent: FlyVisualPredictionAgent,
        path: str | Path,
    ) -> dict[str, Any]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {source}")

        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        vis = agent.visual
        mb = vis.mushroom_body
        cx = vis.central_complex
        pc = vis.predictive_coding

        # Restore Mushroom Body
        mb_data = payload.get("mushroom_body", {})
        if "action_weights" in mb_data and mb_data["action_weights"] is not None:
            w = mb_data["action_weights"]
            if np is not None and hasattr(mb, "_action_weights_np"):
                mb._action_weights_np = np.array(w, dtype=np.float32)
            else:
                mb._action_weights_list = [list(row) for row in w]
        elif "mbon_weights" in mb_data:
            w = mb_data["mbon_weights"]
            if np is not None and hasattr(mb, "_action_weights_np"):
                legacy = np.array(w, dtype=np.float32)
                mb._action_weights_np[1] = np.maximum(legacy, 0.0)
                mb._action_weights_np[2] = np.minimum(legacy, 0.0)
                mb._mbon_weights_np = legacy
            else:
                mb._action_weights_list[1] = [max(value, 0.0) for value in w]
                mb._action_weights_list[2] = [min(value, 0.0) for value in w]
                mb._mbon_weights_np = list(w)
        if "action_weights" in mb_data and mb_data["action_weights"] is not None and "mbon_weights" in mb_data:
            if np is not None and hasattr(mb, "_action_weights_np"):
                mb._mbon_weights_np = np.array(mb_data["mbon_weights"], dtype=np.float32)

        if "kc_history" in mb_data:
            h = mb_data["kc_history"]
            if np is not None and mb._kc_history_np is not None:
                mb._kc_history_np = np.array(h, dtype=np.float32)
            else:
                mb._kc_history_list = list(h)

        # Restore Central Complex
        cx_data = payload.get("central_complex", {})
        if "heading" in cx_data:
            cx._attractor_heading = float(cx_data["heading"])
        if "fast_bias" in cx_data:
            cx._fast_bias = float(cx_data["fast_bias"])
        if "slow_bias" in cx_data:
            cx._slow_bias = float(cx_data["slow_bias"])
        if "arousal" in cx_data:
            cx._arousal = float(cx_data["arousal"])

        # Restore Predictive Coding
        pc_data = payload.get("predictive_coding", {})
        if "current_expectation" in pc_data:
            pc._current_expectation = float(pc_data["current_expectation"])
        if "hypothesis_logits" in pc_data:
            pc._hypothesis_logits = list(pc_data["hypothesis_logits"])

        # Restore Decision Engine
        de_data = payload.get("decision_engine", {})
        if "recent_history" in de_data:
            vis.decision_engine._recent_evidence_history = list(de_data["recent_history"])

        # Restore Attention
        att_data = payload.get("attention", {})
        if "temporal_focus" in att_data:
            vis.attention._temporal_focus = float(att_data["temporal_focus"])
        if "sensory_gain" in att_data:
            vis.attention._sensory_gain = float(att_data["sensory_gain"])

        # Restore Giant Fiber
        gf_data = payload.get("giant_fiber", {})
        if "cooldown" in gf_data:
            vis.giant_fiber._cooldown = int(gf_data["cooldown"])

        # Restore Metabolic Control
        meta_data = payload.get("metabolic_control", {})
        if "energy" in meta_data:
            vis.metabolic_control._energy = float(meta_data["energy"])

        return payload.get("metadata", {})
