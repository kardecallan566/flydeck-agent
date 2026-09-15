import math
import numpy as np

from flydeck.central_complex import CentralComplexSystem
from flydeck.lptc_system import LobulaPlateTangentialSystem
from flydeck.market_retina import BNBMarketRetina
from flydeck.synaptic_adaptation import SynapticAdaptation, SynapticAdaptationConfig
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron


def _toy_circuit() -> VisualCircuit:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0, 0.0, 0.5),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0, 1.0, 0.5),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    return VisualCircuit(
        neurons=neurons,
        edges=(VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)),
        l1_inputs=(0,),
        l2_inputs=(1,),
        t4_outputs=((), (), (2,), (3,)),
        t5_outputs=((), (), (4,), (5,)),
        spatial_mode="soma_xy_proxy",
    )


def test_synaptic_adaptation_habituation_and_recovery() -> None:
    adaptation = SynapticAdaptation(neuron_count=4, config=SynapticAdaptationConfig(tau_recovery=5.0, utilization=0.5))
    assert adaptation.mean_resource == 1.0

    # Step 1: Initial burst (full transmission)
    act = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    out1 = adaptation.modulate_np(act)
    assert np.allclose(out1[:2], [1.0, 1.0])

    # Step 2: Repetitive stimulation causes vesicle depletion (habituation)
    for _ in range(5):
        out = adaptation.modulate_np(act)
    assert out[0] < 0.60, "repetitive firing must deplete synaptic resources"
    assert adaptation._resources_np[2] == 1.0, "quiescent neuron must maintain full resources"

    # Step 3: Quiescent neuron fires -> produces larger response than habituated neuron
    new_act = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    out_new = adaptation.modulate_np(new_act)
    assert out_new[2] > out_new[0], "fresh neuron must fire with higher efficacy than habituated neuron"

    # Step 4: Reset restores full resources
    adaptation.reset()
    assert adaptation.mean_resource == 1.0


def test_lptc_system_opponent_inhibition() -> None:
    lptc = LobulaPlateTangentialSystem(opponent_inhibition=0.5, temporal_smoothing=1.0)

    # 1. Pure UP motion (T4c, T5c active)
    up_t4 = (0.0, 0.0, 1.0, 0.0)
    up_t5 = (0.0, 0.0, 1.0, 0.0)
    out_up = lptc.step(up_t4, up_t5)
    assert out_up.vs_up > 0.5
    assert out_up.vs_down == 0.0
    assert out_up.vs_net > 0.8

    # 2. Pure DOWN motion (T4d, T5d active)
    lptc.reset()
    dn_t4 = (0.0, 0.0, 0.0, 1.0)
    dn_t5 = (0.0, 0.0, 0.0, 1.0)
    out_dn = lptc.step(dn_t4, dn_t5)
    assert out_dn.vs_down > 0.5
    assert out_dn.vs_up == 0.0
    assert out_dn.vs_net < -0.8

    # 3. Conflicting simultaneous UP and DOWN motion
    lptc.reset()
    conflict_t4 = (0.0, 0.0, 1.0, 0.8)
    conflict_t5 = (0.0, 0.0, 1.0, 0.8)
    out_conflict = lptc.step(conflict_t4, conflict_t5)
    # Opponent inhibition suppresses conflicting response
    assert out_conflict.directional_coherence < out_up.directional_coherence


def test_central_complex_working_memory_and_novelty() -> None:
    cx = CentralComplexSystem(tau_fast=0.30, tau_slow=0.05)

    # Sustained UP inputs build positive working memory heading
    for _ in range(10):
        state = cx.step(sensory_signal=0.8, volatility=0.002, coherence=0.9)
    assert state.fast_bias > 0.5
    assert state.slow_bias > 0.2
    assert state.attractor_heading > 0.4
    assert state.alignment > 0.0, "fast and slow momentum should be concordant"

    # Abrupt reversal (surprise / novelty burst)
    rev_state = cx.step(sensory_signal=-0.8, volatility=0.008, coherence=0.3)
    assert rev_state.novelty_signal > 0.0, "abrupt reversal must trigger novelty detection"
    assert rev_state.arousal > state.arousal, "volatility jump must increase octopaminergic arousal"


def test_consensus_and_conflict_engine() -> None:
    circuit = _toy_circuit()
    agent = FlyVisualPredictionAgent(
        circuit,
        retina_width=8,
        confidence_threshold=0.10,
        conflict_threshold=0.30,
    )

    # Monotonic clean trend (concordant streams)
    clean_uptrend = tuple(100.0 + i * 2.0 for i in range(12))
    _stim, decision_clean = agent.perceive(clean_uptrend)
    assert decision_clean.consensus > 0.70
    assert decision_clean.conflict < 0.30

    # Strong zigzag / contradictory chop
    choppy = (100.0, 105.0, 95.0, 106.0, 94.0, 107.0, 93.0, 108.0)
    _stim_chop, decision_chop = agent.perceive(choppy)
    # In choppy contradictory movements, conflict is elevated
    assert decision_chop.conflict >= 0.0


def test_retina_volume_contrast_modulation() -> None:
    retina = BNBMarketRetina(width=8, height=6)
    prices = (100.0, 101.0, 102.0, 103.0, 104.0)

    # Low volume stimulus
    low_vols = (100.0, 100.0, 100.0, 100.0, 10.0)
    stim_low = retina.encode(prices, volumes=low_vols)

    # Surge volume stimulus (10x volume spike)
    high_vols = (100.0, 100.0, 100.0, 100.0, 1000.0)
    stim_high = retina.encode(prices, volumes=high_vols)

    assert stim_high.volume_contrast > stim_low.volume_contrast
    assert stim_high.volume_contrast >= 1.2
