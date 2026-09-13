from flydeck.network import SparseNetwork


def test_positive_reward_changes_selected_path():
    network = SparseNetwork(1, 4, 2, density=1.0, seed=5)
    network.step((1.0,))
    before = [c.weight for c in network._output_connections if c.target == 0]

    network.learn(action=0, reward=1.0, learning_rate=0.1)
    after = [c.weight for c in network._output_connections if c.target == 0]

    assert before != after


def test_negative_reward_moves_selected_path_oppositely():
    network = SparseNetwork(1, 4, 2, density=1.0, seed=5)
    network.step((1.0,))
    before = [c.weight for c in network._output_connections if c.target == 0]

    network.learn(action=0, reward=-1.0, learning_rate=0.1)
    after = [c.weight for c in network._output_connections if c.target == 0]

    assert before != after
