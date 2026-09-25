from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None

from .attention_system import AttentionState, TopDownAttentionModule
from .causal_features import CausalFeatureBank, CausalFeatureVector
from .eligibility import SparseEligibilityTrace
from .episodic_memory import BoundedEpisodicMemory
from .bnb_prediction import Prediction
from .central_complex import CentralComplexState, CentralComplexSystem
from .decision_engine import DecoupledDecision, DecisionReason, DynamicDecisionEngine
from .directional_mechanism import SpatialOffsetDirectionalMechanism
from .giant_fiber import GiantFiberEscapeCircuit, ShockState
from .internal_state import AgentInternalState
from .lptc_system import LPTCOutput, LobulaPlateTangentialSystem
from .market_retina import BNBMarketRetina, RetinaStimulus
from .metabolic_control import MetabolicRiskController, MetabolicState
from .mushroom_body import MushroomBodyAssociativeMemory, MushroomBodyOutput
from .neural_diagnostics import CandleDiagnosticEntry, NeuralDiagnosticsTracer
from .predictive_coding import PredictiveCodingEngine, PredictiveCodingUpdate
from .receptive_fields import ReceptiveField, infer_receptive_fields
from .regime_detector import CausalRegimeDetector
from .temporal_memory import DualTimescaleMemory
from .risk_policy import LightweightRiskPolicy
from .synaptic_adaptation import SynapticAdaptation
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class VisualDecision:
    up_score: float
    down_score: float
    confidence: float
    wait: bool
    consensus: float = 1.0
    conflict: float = 0.0
    arousal: float = 0.20
    fast_bias: float = 0.0
    slow_bias: float = 0.0
    action: Prediction = Prediction.WAIT
    reason: str = "DEFAULT"
    p_wait: float = 0.0
    p_up: float = 0.5
    p_down: float = 0.5
    regime: str = "RANGE"


class MaleCNSVisualSystem:
    """Run the compact MaleCNS motion pathway with Drosophila higher brain centers."""

    def __init__(
        self,
        circuit: VisualCircuit,
        leak: float = 0.30,
        synapse_scale: float = 1.20,
        temporal_gain: float = 0.75,
        micro_steps: int = 4,
        fast_alpha: float = 0.75,
        slow_inhibition_alpha: float = 0.20,
        receptive_field_iterations: int = 12,
        receptive_fields: dict[int, ReceptiveField] | None = None,
        mutual_inhibition_gamma: float = 0.15,
        trend_memory_beta: float = 0.05,
        ablate_spatial: bool = False,
        ablate_temporal: bool = False,
        ablate_slow_inhibition: bool = False,
        ablate_mutual_inhibition: bool = False,
        ablate_t4_t5: bool = False,
        weight_directional: float = 0.25,
        weight_velocity: float = 0.25,
        weight_on_off_balance: float = 0.35,
        weight_trend: float = 0.15,
        ablate_adaptation: bool = False,
        ablate_lptc: bool = False,
        ablate_working_memory: bool = False,
        ablate_neuromodulation: bool = False,
        ablate_conflict_engine: bool = False,
        ablate_mushroom_body: bool = False,
        ablate_predictive_coding: bool = False,
        ablate_attention: bool = False,
        ablate_giant_fiber: bool = False,
        ablate_metabolic: bool = False,
        conflict_threshold: float = 0.35,
    ) -> None:
        if (
            not 0.0 < leak <= 1.0
            or synapse_scale <= 0.0
            or temporal_gain < 0.0
            or micro_steps < 1
            or not 0.0 < fast_alpha <= 1.0
            or not 0.0 < slow_inhibition_alpha <= 1.0
            or receptive_field_iterations < 1
            or not 0.0 <= mutual_inhibition_gamma <= 1.0
            or not 0.0 <= trend_memory_beta <= 1.0
        ):
            raise ValueError("invalid visual dynamics")
        if not circuit.l1_inputs or not circuit.l2_inputs:
            raise ValueError("visual circuit requires L1 and L2 entry neurons")
        if len(circuit.t4_outputs) != 4 or len(circuit.t5_outputs) != 4:
            raise ValueError("visual circuit requires four T4 and four T5 output groups")

        self.circuit = circuit
        self.leak = leak
        self.synapse_scale = synapse_scale
        self.temporal_gain = 0.0 if ablate_temporal else temporal_gain
        self.micro_steps = micro_steps
        self.fast_alpha = fast_alpha
        self.slow_inhibition_alpha = fast_alpha if ablate_slow_inhibition else slow_inhibition_alpha
        self.receptive_field_iterations = receptive_field_iterations
        self.mutual_inhibition_gamma = 0.0 if ablate_mutual_inhibition else mutual_inhibition_gamma
        self.trend_memory_beta = trend_memory_beta
        self.ablate_spatial = ablate_spatial
        self.ablate_temporal = ablate_temporal
        self.ablate_slow_inhibition = ablate_slow_inhibition
        self.ablate_mutual_inhibition = ablate_mutual_inhibition
        self.ablate_t4_t5 = ablate_t4_t5
        self.ablate_adaptation = ablate_adaptation
        self.ablate_lptc = ablate_lptc
        self.ablate_working_memory = ablate_working_memory
        self.ablate_neuromodulation = ablate_neuromodulation
        self.ablate_conflict_engine = ablate_conflict_engine
        self.ablate_mushroom_body = ablate_mushroom_body
        self.ablate_predictive_coding = ablate_predictive_coding
        self.ablate_attention = ablate_attention
        self.ablate_giant_fiber = ablate_giant_fiber
        self.ablate_metabolic = ablate_metabolic
        self.conflict_threshold = conflict_threshold

        self.trend_bias = 0.0
        self.weight_directional = weight_directional
        self.weight_velocity = weight_velocity
        self.weight_on_off_balance = weight_on_off_balance
        self.weight_trend = weight_trend
        self.state = [0.0] * len(circuit.neurons)
        self.filtered_state = [0.0] * len(circuit.neurons)
        self.outgoing: list[list[tuple[int, float]]] = [[] for _ in circuit.neurons]

        # Higher brain systems
        self.synaptic_adaptation = SynapticAdaptation(
            len(circuit.neurons),
            enabled=not ablate_adaptation,
        )
        self.lptc = LobulaPlateTangentialSystem(
            temporal_smoothing=0.60,
            enabled=not ablate_lptc,
        )
        self.central_complex = CentralComplexSystem(
            enabled=not ablate_working_memory,
        )
        self.mushroom_body = MushroomBodyAssociativeMemory(
            input_dim=8,
            kc_count=256,
            enabled=not ablate_mushroom_body,
        )
        self.predictive_coding = PredictiveCodingEngine(
            enabled=not ablate_predictive_coding,
        )
        self.decision_engine = DynamicDecisionEngine(
            max_conflict_tolerance=conflict_threshold,
            enabled=not ablate_conflict_engine,
        )
        self.regime_detector = CausalRegimeDetector()
        self.feature_bank = CausalFeatureBank()
        self.temporal_memory = DualTimescaleMemory()
        self.episodic_memory = BoundedEpisodicMemory()
        self.risk_policy = LightweightRiskPolicy()
        self.eligibility = SparseEligibilityTrace()
        self._previous_policy_action = Prediction.WAIT
        self._previous_policy_vector: tuple[float, ...] = ()
        self._previous_policy_regime = "RANGE"
        self.attention = TopDownAttentionModule(
            enabled=not ablate_attention,
        )
        self.giant_fiber = GiantFiberEscapeCircuit(
            enabled=not ablate_giant_fiber,
        )
        self.metabolic_control = MetabolicRiskController(
            enabled=not ablate_metabolic,
        )
        self.diagnostics = NeuralDiagnosticsTracer()

        # Continuous Agent Internal State flowing across rounds
        self.internal_state = AgentInternalState()
        self.last_decoupled_decision: DecoupledDecision | None = None
        self.last_lptc_out: LPTCOutput | None = None
        self.last_cx_state: CentralComplexState | None = None
        self.last_mb_out: MushroomBodyOutput | None = None
        self.last_pred_update: PredictiveCodingUpdate | None = None
        self.last_shock_state: ShockState | None = None
        self.last_attention_state: AttentionState | None = None

        # Previous step price for causal reinforcement
        self._previous_price: float | None = None
        # Positive bias favours UP and negative bias favours DOWN. It is
        # updated only after the next candle is observed, preventing leakage.
        self.policy_bias = 0.0
        self.policy_bias_learning_rate = 0.04
        self._previous_action_sign = 0
        self.learning_enabled = True

        # Normalize incoming synapse mass
        incoming = [0.0] * len(circuit.neurons)
        for edge in circuit.edges:
            incoming[edge.target] += abs(edge.weight)
        for edge in circuit.edges:
            source_sign = circuit.neurons[edge.source].sign
            total = incoming[edge.target]
            normalized = edge.weight / total if total > 1e-12 else 0.0
            self.outgoing[edge.source].append(
                (edge.target, normalized * source_sign * synapse_scale)
            )

        if np is not None:
            self._has_numpy = True
            self._state_np = np.zeros(len(circuit.neurons), dtype=np.float32)
            self._filtered_state_np = np.zeros(len(circuit.neurons), dtype=np.float32)
            self._alphas_np = np.array([self._alpha(i) for i in range(len(circuit.neurons))], dtype=np.float32)
            self._edge_sources = np.array([e.source for e in circuit.edges], dtype=np.int32)
            self._edge_targets = np.array([e.target for e in circuit.edges], dtype=np.int32)
            source_signs = np.array([circuit.neurons[e.source].sign for e in circuit.edges], dtype=np.float32)
            weights = np.array([e.weight for e in circuit.edges], dtype=np.float32)
            inc = np.array(incoming, dtype=np.float32)[self._edge_targets]
            inc = np.maximum(1e-12, inc)
            self._norm_edge_weights = (weights / inc) * source_signs * synapse_scale
        else:
            self._has_numpy = False

        base_rfs = receptive_fields if receptive_fields is not None else infer_receptive_fields(
            circuit,
            iterations=receptive_field_iterations,
        )
        if ablate_spatial:
            self.receptive_fields = {
                idx: ReceptiveField(
                    x=rf.x,
                    y=rf.y,
                    excitatory_x=rf.x,
                    excitatory_y=rf.y,
                    inhibitory_x=rf.x,
                    inhibitory_y=rf.y,
                    excitatory_mass=rf.excitatory_mass,
                    inhibitory_mass=rf.inhibitory_mass,
                )
                for idx, rf in base_rfs.items()
            }
        else:
            self.receptive_fields = base_rfs

        self.t4_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=self.slow_inhibition_alpha,
        )
        self.t5_directional = SpatialOffsetDirectionalMechanism(
            self.receptive_fields,
            inhibition_alpha=self.slow_inhibition_alpha,
        )
        self.last_directional_t4 = (0.0, 0.0, 0.0, 0.0)
        self.last_directional_t5 = (0.0, 0.0, 0.0, 0.0)

        self.last_stimulus: RetinaStimulus | None = None
        self.last_features: CausalFeatureVector | None = None
        self.last_entry_drive = [0.0] * len(circuit.neurons)
        self.previous_on_field: tuple[tuple[float, ...], ...] | None = None
        self.previous_off_field: tuple[tuple[float, ...], ...] | None = None

    def reset(self, preserve_learning: bool = False) -> None:
        self.state = [0.0] * len(self.state)
        self.filtered_state = [0.0] * len(self.state)
        self.trend_bias = 0.0
        if self._has_numpy:
            self._state_np.fill(0.0)
            self._filtered_state_np.fill(0.0)
        self.last_stimulus = None
        self.last_features = None
        self.last_entry_drive = [0.0] * len(self.state)
        self.previous_on_field = None
        self.previous_off_field = None
        self.t4_directional.reset()
        self.t5_directional.reset()
        self.last_directional_t4 = (0.0, 0.0, 0.0, 0.0)
        self.last_directional_t5 = (0.0, 0.0, 0.0, 0.0)
        self.synaptic_adaptation.reset()
        self.lptc.reset()
        self.central_complex.reset()
        self.mushroom_body.reset(preserve_weights=preserve_learning)
        self.predictive_coding.reset()
        self.decision_engine.reset(preserve_learning=preserve_learning)
        self.attention.reset()
        self.giant_fiber.reset()
        self.metabolic_control.reset()
        self.diagnostics.reset()
        self.internal_state = AgentInternalState()
        self.last_decoupled_decision = None
        self.last_lptc_out = None
        self.last_cx_state = None
        self.last_mb_out = None
        self.last_pred_update = None
        self.last_shock_state = None
        self.last_attention_state = None
        self.regime_detector.reset()
        self.feature_bank.reset()
        self.temporal_memory.reset()
        self.episodic_memory.reset(preserve_memory=preserve_learning)
        self.risk_policy.reset()
        self.eligibility.reset()
        self._previous_policy_action = Prediction.WAIT
        self._previous_policy_vector = ()
        self._previous_policy_regime = "RANGE"
        self._previous_price = None
        # policy_bias is a short-term homeostatic correction, not a learned
        # context association. Never carry it across an episode/split; the
        # action-specific MBON readouts are the persistent learning state.
        self.policy_bias = 0.0
        self._previous_action_sign = 0

    def register_action(self, action: Prediction) -> None:
        """Remember the committed direction for the next causal reward."""
        self._previous_action_sign = (
            1 if action == Prediction.UP else -1 if action == Prediction.DOWN else 0
        )

    def set_learning(self, enabled: bool) -> None:
        """Enable plasticity for training or evaluation."""
        self.learning_enabled = enabled
        self.decision_engine.set_learning(enabled)
        self.episodic_memory.set_learning(enabled)

    def _update_policy_bias(self, observed_return_pct: float) -> None:
        if self._previous_action_sign == 0 or abs(observed_return_pct) < 1e-12:
            return
        outcome_sign = 1 if observed_return_pct > 0 else -1
        error = outcome_sign - self._previous_action_sign
        self.policy_bias = max(
            -0.15,
            min(0.15, self.policy_bias + self.policy_bias_learning_rate * error),
        )

    def step(self, stimulus: RetinaStimulus, current_price: float | None = None,
             features: CausalFeatureVector | None = None) -> tuple[float, ...]:
        # 1. Causal Reinforcement & Shock Detection from prior step
        observed_ret = 0.0
        if current_price is not None and self._previous_price is not None:
            observed_ret = (current_price / self._previous_price - 1.0) * 100.0
            if self.learning_enabled:
                self._update_policy_bias(observed_ret)
                signal = 1.0 if observed_ret > 0.0 else -1.0 if observed_ret < 0.0 else 0.0
                self.mushroom_body.reinforce_actions((0.0, signal, -signal))
                self.decision_engine.observe_outcome(
                    Prediction.UP if observed_ret > 0.0 else Prediction.DOWN if observed_ret < 0.0 else Prediction.WAIT
                )
                self.metabolic_control.update_feedback(observed_ret)
                reward = 1.0 if observed_ret * (1 if self._previous_policy_action == Prediction.UP else -1 if self._previous_policy_action == Prediction.DOWN else 0) > 0 else -1.0
                self.risk_policy.observe(self._previous_policy_action, reward)
                self.eligibility.reinforce(reward)
                self.episodic_memory.add(self._previous_policy_vector, self._previous_policy_action, reward, self._previous_policy_regime)
        if current_price is not None:
            self._previous_price = current_price

        # Giant Fiber regime shock evaluation
        pred_err = self.internal_state.prediction_error
        shock_state = self.giant_fiber.step(
            current_return_pct=observed_ret,
            prediction_error=pred_err,
            volatility_contrast=stimulus.volatility_contrast,
            volume_contrast=stimulus.volume_contrast,
        )
        self.last_shock_state = shock_state

        self.last_stimulus = stimulus
        self.last_features = features
        if features is not None:
            memory_signal = max(-1.0, min(1.0, 0.55 * features.short_return + 0.30 * features.medium_return + 0.15 * features.long_return))
            self.temporal_memory.update(memory_signal, features.novelty)
            active = {index: value for index, value in enumerate(features.values) if abs(value) > 0.25}
            self.eligibility.step(active)
        drive = self._entry_drive(stimulus)
        self.last_entry_drive = drive

        if self._has_numpy:
            drive_np = np.array(drive, dtype=np.float32)
            leak = self.leak
            for _ in range(self.micro_steps):
                self._filtered_state_np += self._alphas_np * (self._state_np - self._filtered_state_np)
                adapted_sources = self.synaptic_adaptation.modulate_np(self._filtered_state_np)
                contributions = adapted_sources[self._edge_sources] * self._norm_edge_weights
                recurrent = np.bincount(self._edge_targets, weights=contributions, minlength=len(self.state))
                target_act = np.maximum(0.0, np.tanh(drive_np + recurrent))
                self._state_np = (1.0 - leak) * self._state_np + leak * target_act
            self.state = self._state_np.tolist()
        else:
            for _ in range(self.micro_steps):
                recurrent = [0.0] * len(self.state)
                for source, outgoing in enumerate(self.outgoing):
                    activity = self.state[source]
                    if activity <= 1e-12:
                        self.filtered_state[source] *= self._alpha(source)
                        continue
                    alpha = self._alpha(source)
                    self.filtered_state[source] += alpha * (activity - self.filtered_state[source])
                    adapted_act = self.synaptic_adaptation.step(self.filtered_state)[source]
                    for target, weight in outgoing:
                        recurrent[target] += adapted_act * weight

                next_state = [0.0] * len(self.state)
                for index in range(len(next_state)):
                    target = max(0.0, math.tanh(drive[index] + recurrent[index]))
                    next_state[index] = (1.0 - self.leak) * self.state[index] + self.leak * target
                self.state = next_state

        self.last_directional_t4 = self.t4_directional.step(
            stimulus,
            self.circuit.t4_outputs,
            polarity="on",
        )
        self.last_directional_t5 = self.t5_directional.step(
            stimulus,
            self.circuit.t5_outputs,
            polarity="off",
        )
        for groups, values in (
            (self.circuit.t4_outputs, self.last_directional_t4),
            (self.circuit.t5_outputs, self.last_directional_t5),
        ):
            for group, value in zip(groups, values):
                for index in group:
                    self.state[index] = value
                    if self._has_numpy:
                        self._state_np[index] = value

        self.last_lptc_out = self.lptc.step(
            self.last_directional_t4,
            self.last_directional_t5,
        )

        self.previous_on_field = stimulus.on_field
        self.previous_off_field = stimulus.off_field
        return tuple(self.state)

    def _alpha(self, index: int) -> float:
        neuron = self.circuit.neurons[index]
        if neuron.sign < 0.0:
            return self.slow_inhibition_alpha
        return self.fast_alpha

    def _entry_drive(self, stimulus: RetinaStimulus) -> list[float]:
        drive = [0.0] * len(self.state)
        # Top-down attention modulates sensory gain
        att_gain = self.internal_state.sensory_gain
        gain = self.temporal_gain * att_gain

        if self.circuit.has_spatial_mapping:
            for neuron in self.circuit.l1_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.on_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_on_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, (current + gain * (current - previous)) * att_gain)
            for neuron in self.circuit.l2_inputs:
                n = self.circuit.neurons[neuron]
                current = _sample_field(stimulus.off_field, n.spatial_x, n.spatial_y)
                previous = _sample_field(self.previous_off_field, n.spatial_x, n.spatial_y)
                drive[neuron] = max(0.0, (current + gain * (current - previous)) * att_gain)
        else:
            on_strength = max((max(row) for row in stimulus.on_field), default=0.0)
            off_strength = max((max(row) for row in stimulus.off_field), default=0.0)
            previous_on = max((max(row) for row in self.previous_on_field), default=0.0)
            previous_off = max((max(row) for row in self.previous_off_field), default=0.0)
            on_strength = max(0.0, (on_strength + gain * (on_strength - previous_on)) * att_gain)
            off_strength = max(0.0, (off_strength + gain * (off_strength - previous_off)) * att_gain)
            for neuron in self.circuit.l1_inputs:
                drive[neuron] = on_strength
            for neuron in self.circuit.l2_inputs:
                drive[neuron] = off_strength
        return drive

    def decision(self, minimum_confidence: float = 0.15) -> VisualDecision:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        st = self.internal_state
        volatility = self.last_stimulus.volatility_contrast if self.last_stimulus else 0.0035
        coherence = self.last_stimulus.coherence if self.last_stimulus else 0.5
        regime_state = self.regime_detector.step(self.last_stimulus, self.last_features) if self.last_stimulus else self.regime_detector.state
        memory_state = self.temporal_memory.state
        if st.feature_vector:
            st.feature_novelty = self.episodic_memory.novelty(st.feature_vector)

        # 1. Top-Down Attention Step
        att_state = self.attention.step(
            arousal=st.arousal,
            prediction_error=st.prediction_error,
            uncertainty=st.uncertainty,
            volatility=volatility,
        )
        self.last_attention_state = att_state
        st.attention_focus = att_state.temporal_focus
        st.sensory_gain = att_state.sensory_gain

        # 2. Motion Stream (LPTC wide-field)
        if self.ablate_t4_t5:
            half = len(self.state) // 2
            dir_up = sum(self.state[:half]) / max(1, half)
            dir_down = sum(self.state[half:]) / max(1, len(self.state) - half)
            dir_tot = dir_up + dir_down
            lptc_signal = (dir_up - dir_down) / dir_tot if dir_tot > 1e-12 else 0.0
        else:
            lptc_signal = self.last_lptc_out.vs_net if self.last_lptc_out else 0.0
            lptc_signal = max(-1.0, min(1.0, lptc_signal + self.policy_bias))

        # 3. Kinematics Stream (Retina velocity and acceleration)
        velocity = self.last_stimulus.velocity if self.last_stimulus else 0.0
        acceleration = self.last_stimulus.acceleration if self.last_stimulus else 0.0
        short_velocity = self.last_stimulus.short_velocity if self.last_stimulus else velocity
        retina_vel_signal = max(-1.0, min(1.0, 0.65 * velocity + 0.35 * short_velocity + 0.25 * acceleration))

        # 4. Perceptual Balance (L1 vs L2)
        l1_act = sum(self.state[n] for n in self.circuit.l1_inputs) / max(1, len(self.circuit.l1_inputs))
        l2_act = sum(self.state[n] for n in self.circuit.l2_inputs) / max(1, len(self.circuit.l2_inputs))
        bal_tot = l1_act + l2_act
        balance_signal = (l1_act - l2_act) / bal_tot if bal_tot > 1e-12 else 0.0

        # 5. Central Complex Integration (Recurrent feedback from MB valence + Giant Fiber reset)
        mb_prior_valence = self.last_mb_out.valence if self.last_mb_out else 0.0
        is_shock = self.last_shock_state.is_shock if self.last_shock_state else False
        st.is_shock = is_shock
        st.shock_magnitude = self.last_shock_state.shock_magnitude if self.last_shock_state else 0.0

        cx_state = self.central_complex.step(
            sensory_signal=0.40 * lptc_signal + 0.35 * retina_vel_signal + 0.25 * memory_state.fast,
            volatility=volatility,
            coherence=coherence,
            mb_feedback=mb_prior_valence,
            reset_heading=is_shock,
        )
        self.last_cx_state = cx_state

        # 6. Mushroom Body Sparse Associative Memory (Zero-Centered Symmetric Context)
        norm_coherence = (coherence - 0.5) * 2.0
        norm_volatility = min(1.0, max(-1.0, (volatility - 0.0035) * 200.0))
        context_vector = (
            lptc_signal,
            retina_vel_signal,
            balance_signal,
            cx_state.attractor_heading,
            cx_state.fast_bias,
            cx_state.slow_bias,
            norm_volatility,
            norm_coherence,
        )
        mb_out = self.mushroom_body.perceive(context_vector)
        self.last_mb_out = mb_out

        # 7. Predictive Coding Loop & Hypothesis Competition
        pred_update = self.predictive_coding.step(
            observed_motion=lptc_signal,
            cx_context=cx_state.attractor_heading,
            mb_valence=mb_out.valence,
            volatility=volatility,
        )
        self.last_pred_update = pred_update

        # 8. Metabolic Risk Step
        meta_state = self.metabolic_control.step()
        st.metabolic_energy = meta_state.energy_level

        # 9. Synthesize Continuous Internal State
        st.perceptual_balance = balance_signal
        st.policy_bias = self.policy_bias
        st.retina_velocity = retina_vel_signal
        st.retina_acceleration = acceleration
        st.volume_contrast = self.last_stimulus.volume_contrast if self.last_stimulus else 1.0
        st.volatility_contrast = volatility
        st.coherence = coherence
        st.regime = regime_state.regime.value
        st.regime_confidence = max(regime_state.probabilities)
        st.regime_probabilities = regime_state.probabilities
        st.regime_duration = regime_state.duration
        st.feature_vector = self.last_features.values if self.last_features else ()
        st.fast_memory = memory_state.fast
        st.slow_memory = memory_state.slow
        st.feature_novelty = memory_state.novelty
        st.vs_net = lptc_signal
        st.hs_net = self.last_lptc_out.hs_net if self.last_lptc_out else 0.0
        st.motion_energy = self.last_lptc_out.motion_energy if self.last_lptc_out else 0.0
        st.cx_fast_bias = cx_state.fast_bias
        st.cx_slow_bias = cx_state.slow_bias
        st.cx_heading = cx_state.attractor_heading
        st.mb_valence = mb_out.valence
        st.mb_novelty = mb_out.novelty
        st.mb_action_values = mb_out.action_values
        st.expectation = pred_update.expectation
        st.prediction_error = pred_update.prediction_error
        st.signed_error = pred_update.signed_error
        st.arousal = max(cx_state.arousal, pred_update.dopamine_burst)
        st.dopamine_burst = pred_update.dopamine_burst
        st.uncertainty = pred_update.uncertainty
        st.hypothesis_probs = pred_update.hypothesis_probs

        # 10. Decoupled Action Selection via DynamicDecisionEngine with Metabolic Modifier
        decision = self.decision_engine.decide(
            st,
            minimum_confidence=minimum_confidence,
            metabolic_modifier=meta_state.threshold_modifier,
        )
        risk_action = self.risk_policy.before_action(decision.action, st.feature_novelty)
        if risk_action != decision.action:
            decision = DecoupledDecision(
                action=Prediction.WAIT,
                reason=DecisionReason.WAIT_RISK_POLICY,
                up_score=decision.up_score,
                down_score=decision.down_score,
                confidence=decision.confidence,
                wait=True,
                conflict=decision.conflict,
                uncertainty=decision.uncertainty,
                temporal_consistency=decision.temporal_consistency,
                p_wait=decision.p_wait,
                p_up=decision.p_up,
                p_down=decision.p_down,
                regime=decision.regime,
            )
        self.last_decoupled_decision = decision
        st.conflict = decision.conflict
        st.temporal_consistency = decision.temporal_consistency
        self._previous_policy_action = decision.action
        self._previous_policy_vector = st.feature_vector
        self._previous_policy_regime = st.regime

        return VisualDecision(
            up_score=decision.up_score,
            down_score=decision.down_score,
            confidence=decision.confidence,
            wait=decision.wait,
            consensus=1.0 - decision.conflict,
            conflict=decision.conflict,
            arousal=st.arousal,
            fast_bias=st.cx_fast_bias,
            slow_bias=st.cx_slow_bias,
            action=decision.action,
            reason=decision.reason.value,
            p_wait=decision.p_wait,
            p_up=decision.p_up,
            p_down=decision.p_down,
            regime=decision.regime,
        )

    def entry_activity_grid(self, width: int = 32, height: int = 16) -> tuple[tuple[float, ...], ...]:
        if width < 2 or height < 2:
            raise ValueError("grid dimensions are too small")
        grid = [[0.0] * width for _ in range(height)]
        counts = [[0] * width for _ in range(height)]
        for neuron_index in self.circuit.l1_inputs + self.circuit.l2_inputs:
            neuron = self.circuit.neurons[neuron_index]
            if neuron.spatial_x is None or neuron.spatial_y is None:
                continue
            x = min(width - 1, max(0, int(round(neuron.spatial_x * (width - 1)))))
            y = min(height - 1, max(0, int(round(neuron.spatial_y * (height - 1)))))
            grid[y][x] += self.state[neuron_index]
            counts[y][x] += 1
        return tuple(tuple(value / max(1, counts[y][x]) for x, value in enumerate(row)) for y, row in enumerate(grid))

    def _directional_activity(self, groups: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        return tuple(sum(self.state[index] for index in group) / max(1, len(group)) for group in groups)


class FlyVisualPredictionAgent:
    """BNB prediction agent with integrated closed-loop Drosophila brain systems."""

    def __init__(
        self,
        circuit: VisualCircuit,
        retina_width: int = 32,
        retina_height: int = 16,
        confidence_threshold: float = 0.15,
        receptive_fields: dict[int, ReceptiveField] | None = None,
        mutual_inhibition_gamma: float = 0.15,
        trend_memory_beta: float = 0.05,
        ablate_spatial: bool = False,
        ablate_temporal: bool = False,
        ablate_slow_inhibition: bool = False,
        ablate_mutual_inhibition: bool = False,
        ablate_t4_t5: bool = False,
        weight_directional: float = 0.25,
        weight_velocity: float = 0.25,
        weight_on_off_balance: float = 0.35,
        weight_trend: float = 0.15,
        ablate_adaptation: bool = False,
        ablate_lptc: bool = False,
        ablate_working_memory: bool = False,
        ablate_neuromodulation: bool = False,
        ablate_conflict_engine: bool = False,
        ablate_mushroom_body: bool = False,
        ablate_predictive_coding: bool = False,
        ablate_attention: bool = False,
        ablate_giant_fiber: bool = False,
        ablate_metabolic: bool = False,
        conflict_threshold: float = 0.35,
    ) -> None:
        self.retina = BNBMarketRetina(retina_width, retina_height)
        self.visual = MaleCNSVisualSystem(
            circuit,
            receptive_fields=receptive_fields,
            mutual_inhibition_gamma=mutual_inhibition_gamma,
            trend_memory_beta=trend_memory_beta,
            ablate_spatial=ablate_spatial,
            ablate_temporal=ablate_temporal,
            ablate_slow_inhibition=ablate_slow_inhibition,
            ablate_mutual_inhibition=ablate_mutual_inhibition,
            ablate_t4_t5=ablate_t4_t5,
            weight_directional=weight_directional,
            weight_velocity=weight_velocity,
            weight_on_off_balance=weight_on_off_balance,
            weight_trend=weight_trend,
            ablate_adaptation=ablate_adaptation,
            ablate_lptc=ablate_lptc,
            ablate_working_memory=ablate_working_memory,
            ablate_neuromodulation=ablate_neuromodulation,
            ablate_conflict_engine=ablate_conflict_engine,
            ablate_mushroom_body=ablate_mushroom_body,
            ablate_predictive_coding=ablate_predictive_coding,
            ablate_attention=ablate_attention,
            ablate_giant_fiber=ablate_giant_fiber,
            ablate_metabolic=ablate_metabolic,
            conflict_threshold=conflict_threshold,
        )
        self.confidence_threshold = confidence_threshold

    def reset(self, preserve_learning: bool = False) -> None:
        self.visual.reset(preserve_learning=preserve_learning)

    def commit_action(self, action: Prediction) -> None:
        """Commit a training-time action for the next causal reward signal."""
        self.visual.register_action(action)

    def set_learning(self, enabled: bool) -> None:
        self.visual.set_learning(enabled)

    def perceive(
        self,
        prices: tuple[float, ...],
        volumes: tuple[float, ...] | None = None,
    ) -> tuple[RetinaStimulus, VisualDecision]:
        current_price = prices[-1] if prices else None
        features = self.visual.feature_bank.transform(prices, volumes=volumes)
        stimulus = self.retina.encode(prices, volumes=volumes)
        self.visual.step(stimulus, current_price=current_price, features=features)
        decision = self.visual.decision(self.confidence_threshold)
        self.visual.register_action(decision.action)
        return stimulus, decision

    @property
    def internal_state(self) -> AgentInternalState:
        return self.visual.internal_state

    @property
    def diagnostics(self) -> NeuralDiagnosticsTracer:
        return self.visual.diagnostics

    @property
    def neuron_count(self) -> int:
        return len(self.visual.circuit.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.visual.circuit.edges)


def _sample_field(field: tuple[tuple[float, ...], ...] | None, spatial_x: float | None, spatial_y: float | None) -> float:
    if spatial_x is None or spatial_y is None or not field or not field[0]:
        return 0.0
    height = len(field)
    width = len(field[0])
    x = min(width - 1, max(0.0, spatial_x * (width - 1)))
    y = min(height - 1, max(0.0, spatial_y * (height - 1)))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(width - 1, x0 + 1), min(height - 1, y0 + 1)
    fx, fy = x - x0, y - y0
    return field[y0][x0] * (1.0 - fx) * (1.0 - fy) + field[y0][x1] * fx * (1.0 - fy) + field[y1][x0] * (1.0 - fx) * fy + field[y1][x1] * fx * fy
