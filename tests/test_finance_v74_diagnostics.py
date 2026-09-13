from flydeck.agent import Agent
from flydeck.finance import CryptoTradingEnvironment
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_v74_diagnostics import diagnose_finance_agent, format_comprehensive_diagnostics


def make_agent(seed=42):
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    env = CryptoTradingEnvironment([], max_steps=1) if False else None
    return Agent(observation_size=encoder.output_size, action_size=3, hidden_size=32, density=0.10, learning_rate=0.005, seed=seed)


def test_comprehensive_diagnostics_is_non_destructive():
    agent = make_agent()
    before = tuple(c.weight for c in agent.network._output_connections)
    report = diagnose_finance_agent(agent, training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    after = tuple(c.weight for c in agent.network._output_connections)
    assert before == after
    assert report.network.total_connections == agent.network.connection_count
    assert report.encoder.unique_sparse_states > 0
    assert report.action_diagnostics[0].count + report.action_diagnostics[1].count + report.action_diagnostics[2].count == 64


def test_diagnostics_covers_all_major_layers():
    report = diagnose_finance_agent(make_agent(), training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    assert len(report.feature_diagnostics) == 12
    assert len(report.action_diagnostics) == 3
    assert report.network.input_connections > 0
    assert report.network.recurrent_connections > 0
    assert report.network.output_connections > 0
    assert report.environment.steps == 64
    assert report.recurrence.sequence_steps == 64
    assert report.generalization.market_count == 2
    assert report.suspects


def test_formatter_contains_sections_and_root_cause_summary():
    report = diagnose_finance_agent(make_agent(), training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    text = format_comprehensive_diagnostics(report)
    for section in (
        "DATA / OBSERVATION PREDICTIVENESS",
        "ACTION / VALUE HEAD",
        "ENVIRONMENT / INCENTIVE AUDIT",
        "ENCODER / REPRESENTATION",
        "RECURRENT CIRCUIT",
        "TOPOLOGY / WEIGHTS",
        "GENERALIZATION",
        "SUSPECTED ROOT CAUSES",
    ):
        assert section in text
