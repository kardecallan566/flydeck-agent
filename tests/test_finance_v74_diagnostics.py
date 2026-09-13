from flydeck.agent import Agent
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_v74_diagnostics import diagnose_finance_agent, format_diagnostic_report


def make_agent(seed=42):
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    return Agent(observation_size=encoder.output_size, action_size=3, hidden_size=32, density=0.10, learning_rate=0.005, seed=seed)


def test_comprehensive_diagnostics_is_non_destructive():
    agent = make_agent()
    before = tuple(c.weight for c in agent.network._output_connections)
    report = diagnose_finance_agent(agent, training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    after = tuple(c.weight for c in agent.network._output_connections)
    assert before == after
    assert report.sparse_unique_states > 0
    assert sum(report.action_counts) == 64
    assert report.states == 64


def test_diagnostics_covers_all_major_layers():
    report = diagnose_finance_agent(make_agent(), training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    assert len(report.feature_stats) == 12
    assert len(report.score_distributions) == 3
    assert len(report.advantage_distributions) == 3
    assert len(report.immediate_reward_distributions) == 3
    assert report.inferred_regimes
    assert report.recurrent_effect.mean >= 0.0
    assert report.input_weight.minimum <= report.input_weight.maximum
    assert report.recurrent_weight.minimum <= report.recurrent_weight.maximum
    assert report.output_weight.minimum <= report.output_weight.maximum
    assert report.unseen_action_pct[0] + report.unseen_action_pct[1] + report.unseen_action_pct[2] > 99.9


def test_formatter_contains_every_diagnostic_section():
    report = diagnose_finance_agent(make_agent(), training_seeds=(100,), unseen_seeds=(10_000,), market_length=64, max_steps=32)
    text = format_diagnostic_report(report)
    for section in (
        "LEARNED ACTION VALUES",
        "ENVIRONMENT COUNTERFACTUALS",
        "MARKET SIGNAL / 12 FEATURES",
        "SPARSE ENCODER",
        "RECURRENT NETWORK",
        "TOPOLOGY / WEIGHTS",
        "GENERALIZATION",
        "FINDINGS",
        "SUSPECTED ROOT CAUSES",
    ):
        assert section in text
