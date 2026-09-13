from flydeck.agent import Agent
from flydeck.finance_v8 import train_synthetic_crypto_v8
from flydeck.network import SparseNetwork


def test_selected_action_td_update_does_not_push_competitors_down():
    network = SparseNetwork(input_size=2, hidden_size=4, output_size=3, density=1.0, seed=42)
    network.step((1.0, -0.5))
    before = [connection.weight for connection in network._output_connections]

    network.learn_td_selected_action(
        action=0,
        reward=1.0,
        next_scores=(0.0, 0.0, 0.0),
        done=True,
        learning_rate=0.01,
    )

    for old, new, connection in zip(before, network._output_connections, network._output_connections):
        if connection.target != 0:
            assert new.weight == old


def test_selected_action_td_update_changes_selected_action_connections():
    network = SparseNetwork(input_size=2, hidden_size=4, output_size=3, density=1.0, seed=42)
    network.step((1.0, -0.5))
    before = [connection.weight for connection in network._output_connections]

    network.learn_td_selected_action(
        action=1,
        reward=1.0,
        next_scores=(0.0, 0.0, 0.0),
        done=True,
        learning_rate=0.01,
    )

    changed = [
        connection.target == 1 and connection.weight != old
        for old, connection in zip(before, network._output_connections)
    ]
    assert any(changed)


def test_v8_training_runs_with_selected_action_update():
    agent = Agent(
        observation_size=27,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )
    result = train_synthetic_crypto_v8(
        agent,
        episodes=3,
        market_length=64,
        max_steps=30,
        seed=100,
        epsilon=0.20,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    )
    assert result.total_actions == 90
    assert result.total_trades >= 0
    assert 0.0 <= result.trade_frequency <= 1.0
    assert result.average_prediction_error >= 0.0
