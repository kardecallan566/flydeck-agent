from flydeck.market_retina import BNBMarketRetina
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit, VisualEdge, VisualNeuron


def test_market_retina_is_causal_and_has_visual_field() -> None:
    retina = BNBMarketRetina(width=8, height=6)
    stimulus = retina.encode((100.0, 101.0, 102.0, 104.0, 103.0))
    assert len(stimulus.on_field) == 6
    assert len(stimulus.on_field[0]) == 8
    assert len(stimulus.off_field) == 6
    assert stimulus.velocity != 0.0
    assert len(stimulus.directions) == 4
    assert "#" in retina.ascii(stimulus) or "o" in retina.ascii(stimulus)


def test_visual_circuit_round_trip(tmp_path) -> None:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    circuit = VisualCircuit(
        neurons=neurons,
        edges=(VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)),
        on_inputs=((0,), (), (), ()),
        off_inputs=((1,), (), (), ()),
        t4_outputs=((), (), (2,), (3,)),
        t5_outputs=((), (), (4,), (5,)),
    )
    path = tmp_path / "visual.json"
    circuit.save(path)
    loaded = VisualCircuit.load(path)
    assert loaded.neurons == neurons
    assert loaded.edges == circuit.edges


def test_visual_agent_returns_valid_decision() -> None:
    neurons = (
        VisualNeuron(1, "L1", "visual_entry", "acetylcholine", 1.0),
        VisualNeuron(2, "L2", "visual_entry", "acetylcholine", 1.0),
        VisualNeuron(3, "T4c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(4, "T4d", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(5, "T5c", "motion_detector", "acetylcholine", 1.0),
        VisualNeuron(6, "T5d", "motion_detector", "acetylcholine", 1.0),
    )
    circuit = VisualCircuit(neurons, (VisualEdge(0, 2, 1.0), VisualEdge(1, 3, 1.0)), ((0,), (), (), ()), ((1,), (), (), ()), ((), (), (2,), (3,)), ((), (), (4,), (5,)))
    agent = FlyVisualPredictionAgent(circuit, retina_width=8, retina_height=6)
    _stimulus, decision = agent.perceive((100.0, 100.5, 101.0, 102.0, 103.0))
    assert decision.up_score >= 0.0
    assert decision.down_score >= 0.0
    assert 0.0 <= decision.confidence <= 1.0
