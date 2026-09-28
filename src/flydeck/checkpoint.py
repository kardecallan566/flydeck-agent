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

from .bnb_prediction import Prediction
from .episodic_memory import Episode
from .internal_state import AgentInternalState
from .risk_policy import RiskState
from .temporal_memory import TemporalMemoryState
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

        state_dict = asdict(vis.internal_state)

        payload: dict[str, Any] = {
            "version": 2,
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
            "runtime": {
                "previous_price": vis._previous_price,
                "policy_bias": vis.policy_bias,
                "previous_action_sign": vis._previous_action_sign,
                "previous_policy_action": int(vis._previous_policy_action),
                "previous_policy_vector": list(vis._previous_policy_vector),
                "previous_policy_regime": vis._previous_policy_regime,
                "learning_enabled": vis.learning_enabled,
            },
            "risk_policy": asdict(vis.risk_policy.state),
            "eligibility": vis.eligibility.values,
            "episodic_memory": [
                {
                    "vector": list(episode.vector),
                    "action": int(episode.action),
                    "reward": episode.reward,
                    "regime": episode.regime,
                }
                for episode in vis.episodic_memory._episodes
            ],
            "temporal_memory": asdict(vis.temporal_memory.state),
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

        # Version 2 restores the causal runtime cursor needed to continue
        # learning after a process restart instead of silently starting a new
        # episode with old weights.
        runtime = payload.get("runtime", {})
        if runtime:
            previous_price = runtime.get("previous_price")
            vis._previous_price = float(previous_price) if previous_price is not None else None
            vis.policy_bias = float(runtime.get("policy_bias", vis.policy_bias))
            vis._previous_action_sign = int(runtime.get("previous_action_sign", 0))
            vis._previous_policy_action = Prediction(int(runtime.get("previous_policy_action", 0)))
            vis._previous_policy_vector = tuple(float(v) for v in runtime.get("previous_policy_vector", ()))
            vis._previous_policy_regime = str(runtime.get("previous_policy_regime", "RANGE"))
            vis.set_learning(bool(runtime.get("learning_enabled", True)))

        risk_data = payload.get("risk_policy")
        if isinstance(risk_data, dict):
            vis.risk_policy._state = RiskState(
                consecutive_losses=int(risk_data.get("consecutive_losses", 0)),
                cooldown=int(risk_data.get("cooldown", 0)),
                drawdown=float(risk_data.get("drawdown", 0.0)),
                novelty=float(risk_data.get("novelty", 0.0)),
                risk_modifier=float(risk_data.get("risk_modifier", 0.0)),
            )

        eligibility_data = payload.get("eligibility")
        if isinstance(eligibility_data, dict):
            vis.eligibility._values = {
                int(key): float(value) for key, value in eligibility_data.items()
            }

        episodes = payload.get("episodic_memory")
        if isinstance(episodes, list):
            vis.episodic_memory._episodes = [
                Episode(
                    vector=tuple(float(value) for value in row.get("vector", ())),
                    action=Prediction(int(row.get("action", 0))),
                    reward=float(row.get("reward", 0.0)),
                    regime=str(row.get("regime", "RANGE")),
                )
                for row in episodes
                if isinstance(row, dict)
            ][-vis.episodic_memory.capacity :]

        temporal_data = payload.get("temporal_memory")
        if isinstance(temporal_data, dict):
            vis.temporal_memory._state = TemporalMemoryState(
                fast=float(temporal_data.get("fast", 0.0)),
                slow=float(temporal_data.get("slow", 0.0)),
                novelty=float(temporal_data.get("novelty", 0.0)),
            )

        internal_data = payload.get("internal_state")
        if isinstance(internal_data, dict):
            tuple_fields = {
                "mb_action_values",
                "regime_probabilities",
                "feature_vector",
                "hypothesis_probs",
            }
            normalized = {
                key: tuple(value) if key in tuple_fields and isinstance(value, list) else value
                for key, value in internal_data.items()
            }
            allowed = AgentInternalState.__dataclass_fields__
            vis.internal_state = AgentInternalState(
                **{key: value for key, value in normalized.items() if key in allowed}
            )

        return payload.get("metadata", {})
