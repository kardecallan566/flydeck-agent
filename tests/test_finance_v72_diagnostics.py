from pytest import approx

from flydeck.agent import Agent
from flydeck.finance import SyntheticCryptoMarket
from flydeck.finance_v7_diagnostics import aggregate_diagnostics, diagnose_policy, format_diagnostics


def _agent() -> Agent:
    return Agent(
        observation_size=27,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )


def test_v72_diagnostics_collects_policy_and_sparse_state_metrics():
    result = diagnose_policy(
        _agent(),
        SyntheticCryptoMarket(length=64, seed=100).generate(),
        label="test",
        max_steps=30,
    )

    assert result.total_actions == 30
    assert sum(result.actions) == 30
    assert len(result.average_scores) == 3
    assert result.average_score_spread >= 0.0
    assert result.unique_sparse_states >= 1
    assert result.average_active_magnitude > 0.0
    assert set(result.inferred_regimes) == {
        "trend_up", "trend_down", "sideways", "volatile", "reversal"
    }
    assert sum(stats.total for stats in result.inferred_regimes.values()) == 30


def test_v72_diagnostics_aggregate_markets():
    results = [
        diagnose_policy(
            _agent(),
            SyntheticCryptoMarket(length=64, seed=seed).generate(),
            label=str(seed),
            max_steps=20,
        )
        for seed in (100, 101)
    ]
    aggregate = aggregate_diagnostics(results, "aggregate")

    assert aggregate.total_actions == 40
    assert aggregate.unique_sparse_states >= max(result.unique_sparse_states for result in results)
    assert aggregate.average_score_spread >= 0.0
    assert aggregate.average_active_magnitude > 0.0
    assert aggregate.inferred_regimes["volatile"].total >= 0


def test_v72_format_contains_key_diagnostic_sections():
    result = diagnose_policy(
        _agent(),
        SyntheticCryptoMarket(length=64, seed=100).generate(),
        label="test",
        max_steps=10,
    )
    text = format_diagnostics(result)
    assert "actions:" in text
    assert "average scores:" in text
    assert "unique sparse states:" in text
    assert "inferred regimes" in text
    assert "HOLD" in text
    assert "BUY" in text
    assert "SELL" in text
