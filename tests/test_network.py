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
