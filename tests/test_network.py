from flydeck.network import SparseNetwork


def test_network_is_deterministic_for_same_seed():
    first = SparseNetwork(3, 8, 2, density=0.25, seed=7)
    second = SparseNetwork(3, 8, 2, density=0.25, seed=7)

    assert first.connection_count == second.connection_count
    assert first.step((1.0, 0.0, -1.0)) == second.step((1.0, 0.0, -1.0))


def test_network_reuses_internal_state():
    network = SparseNetwork(1, 4, 2, density=1.0, seed=1)
    first = network.step((1.0,))
    second = network.step((0.0,))

    assert first != second


def test_reset_clears_state():
    network = SparseNetwork(1, 4, 2, density=1.0, seed=1)
    network.step((1.0,))
    network.reset()

    reset_output = network.step((0.0,))

    fresh = SparseNetwork(1, 4, 2, density=1.0, seed=1)
    assert reset_output == fresh.step((0.0,))


def test_td_learning_is_deterministic_and_resettable():
    first = SparseNetwork(2, 8, 3, density=0.25, seed=9)
    second = SparseNetwork(2, 8, 3, density=0.25, seed=9)
    first_scores = first.step((1.0, -0.5))
    second_scores = second.step((1.0, -0.5))
    first_error = first.learn_td(1, 0.2, (0.1, 0.4, -0.2), False)
    second_error = second.learn_td(1, 0.2, (0.1, 0.4, -0.2), False)

    assert first_scores == second_scores
    assert first_error == second_error
    assert first.step((0.25, 0.5)) == second.step((0.25, 0.5))

    first.reset()
    fresh = SparseNetwork(2, 8, 3, density=0.25, seed=9)
    assert first.step((0.0, 0.0)) == fresh.step((0.0, 0.0))
