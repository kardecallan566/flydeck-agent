from __future__ import annotations

from flydeck.agent import Agent
from flydeck.finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_v83 import train_synthetic_crypto_v83


def _agent(seed: int = 42) -> Agent:
    candles = SyntheticCryptoMarket(length=64, seed=7).generate()
    environment = CryptoTradingEnvironment(candles, max_steps=20)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    return Agent(
        observation_size=encoder.output_size,
        action_size=environment.action_size,
        hidden_size=16,
        density=0.20,
        learning_rate=0.01,
        seed=seed,
    )


def test_all_action_td_updates_each_action_without_competitive_pushdown():
    agent = _agent()
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    observation = (0.1,) * 12
    agent.network.reset()
    scores = agent.observe(encoder.encode(observation))
    before = tuple(connection.weight for connection in agent.network._output_connections)
    decision_state = tuple(agent.network._decision_state)

    expected_errors = []
    for action, reward in enumerate((0.8, -0.4, 0.2)):
        current = sum(
            decision_state[connection.source] * connection.weight
            for connection in before
            if connection.target == action
        )
        expected_errors.append(max(-1.0, min(1.0, reward - current)))

    td_errors = agent.network.learn_td_all_actions(
        rewards=(0.8, -0.4, 0.2),
        next_scores_by_action=((0.0, 0.0, 0.0),) * 3,
        done_by_action=(True, True, True),
        learning_rate=0.01,
        discount=0.97,
        trace_decay=0.0,
    )

    assert td_errors == tuple(expected_errors)
    changed_targets = {
        connection.target
        for old, connection in zip(before, agent.network._output_connections)
        if old != connection.weight
    }
    assert changed_targets == {0, 1, 2}
    assert scores != ()


def test_all_action_td_is_deterministic():
    def run(seed: int):
        agent = _agent(seed)
        encoder = SparseMarketEncoder(feature_count=12, winners=4)
        agent.network.reset()
        agent.observe(encoder.encode((0.1,) * 12))
        errors = agent.network.learn_td_all_actions(
            rewards=(0.2, -0.1, 0.05),
            next_scores_by_action=((0.3, 0.1, -0.2), (0.1, 0.4, -0.1), (0.2, -0.2, 0.5)),
            done_by_action=(False, False, False),
            learning_rate=0.01,
            discount=0.97,
            trace_decay=0.85,
        )
        weights = tuple(connection.weight for connection in agent.network._output_connections)
        return errors, weights

    assert run(11) == run(11)


def test_synthetic_v83_training_runs():
    result = train_synthetic_crypto_v83(
        _agent(),
        episodes=2,
        market_length=64,
        max_steps=20,
    )
    assert result.episodes == 2
    assert result.total_actions == 40
    assert result.total_trades >= 0
    assert result.average_prediction_error >= 0.0
