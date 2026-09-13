from flydeck.agent import Agent
from flydeck.finance import SyntheticCryptoMarket
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_v82 import collect_position_aware_diagnostics


def _candles():
    return SyntheticCryptoMarket(length=96, seed=321).generate()


def _agent(seed=42):
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    return Agent(
        observation_size=encoder.output_size,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=seed,
    )


def test_position_diagnostics_track_exact_pre_action_counterfactuals():
    diagnostics = collect_position_aware_diagnostics(_agent(), _candles(), max_steps=40)
    assert len(diagnostics.records) == 40
    assert all(len(record.counterfactual_rewards) == 3 for record in diagnostics.records)
    assert diagnostics.flat_records + diagnostics.long_records == 40
    assert sum(sum(row) for row in diagnostics.action_state_transitions) == 39
    assert sum(sum(row) for row in diagnostics.position_state_transitions) == 40


def test_position_diagnostics_expose_portfolio_outcomes_at_horizons():
    diagnostics = collect_position_aware_diagnostics(_agent(), _candles(), max_steps=40)
    assert len(diagnostics.average_realized_portfolio_returns_by_action) == 3
    assert all(len(values) == 5 for values in diagnostics.average_realized_portfolio_returns_by_action)
    assert len(diagnostics.average_realized_portfolio_returns_by_position_state) == 2
    assert all(len(values) == 5 for values in diagnostics.average_realized_portfolio_returns_by_position_state)
    assert all(len(record.realized_portfolio_returns) > 0 for record in diagnostics.records[:16])


def test_position_diagnostics_does_not_update_network_weights():
    agent = _agent()
    before_input = tuple(connection.weight for connection in agent.network._input_connections)
    before_recurrent = tuple(connection.weight for connection in agent.network._recurrent_connections)
    before_output = tuple(connection.weight for connection in agent.network._output_connections)
    collect_position_aware_diagnostics(agent, _candles(), max_steps=20)
    assert before_input == tuple(connection.weight for connection in agent.network._input_connections)
    assert before_recurrent == tuple(connection.weight for connection in agent.network._recurrent_connections)
    assert before_output == tuple(connection.weight for connection in agent.network._output_connections)


def test_counterfactuals_share_the_same_pre_action_state():
    diagnostics = collect_position_aware_diagnostics(_agent(), _candles(), max_steps=12)
    for record in diagnostics.records:
        assert record.position_before >= 0.0
        assert record.position_after >= 0.0
        assert len(record.counterfactual_rewards) == 3
        assert record.portfolio_value_before > 0.0
        assert record.portfolio_value_after > 0.0
