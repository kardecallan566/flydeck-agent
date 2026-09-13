from flydeck.network import SparseNetwork


def _weights(network: SparseNetwork) -> tuple[float, ...]:
    return tuple(connection.weight for connection in (
        *network._input_connections,
        *network._recurrent_connections,
        *network._output_connections,
    ))


def _train_step(network: SparseNetwork) -> None:
    network.step((0.4, -0.2, 0.7))
    network.step((0.2, 0.3, -0.5))
    network.learn_td(action=1, reward=0.4, next_scores=(0.1, 0.3, -0.1), done=False)


def test_internal_plasticity_is_disabled_by_default():
    network = SparseNetwork(3, 4, 3, density=1.0, seed=7)
    before = _weights(network)
    _train_step(network)
    after = _weights(network)

    # Output learning remains active, but internal weights are unchanged.
    input_count = len(network._input_connections)
    recurrent_count = len(network._recurrent_connections)
    assert before[: input_count + recurrent_count] == after[: input_count + recurrent_count]
    assert before[input_count + recurrent_count :] != after[input_count + recurrent_count :]


def test_internal_plasticity_updates_input_and_recurrent_weights():
    network = SparseNetwork(
        3, 4, 3, density=1.0, seed=7,
        input_plasticity=0.01,
        recurrent_plasticity=0.05,
    )
    before_input = tuple(c.weight for c in network._input_connections)
    before_recurrent = tuple(c.weight for c in network._recurrent_connections)
    _train_step(network)
    after_input = tuple(c.weight for c in network._input_connections)
    after_recurrent = tuple(c.weight for c in network._recurrent_connections)

    assert before_input != after_input
    assert before_recurrent != after_recurrent


def test_plasticity_preserves_sparse_topology_and_weight_bounds():
    network = SparseNetwork(
        8, 16, 3, density=0.10, seed=42,
        input_plasticity=0.01,
        recurrent_plasticity=0.05,
    )
    counts = (len(network._input_connections), len(network._recurrent_connections), len(network._output_connections))
    for _ in range(20):
        network.step(tuple(0.1 * ((index % 5) - 2) for index in range(8)))
        network.learn_td(1, 0.5, (0.2, 0.4, 0.1), False)

    assert counts == (len(network._input_connections), len(network._recurrent_connections), len(network._output_connections))
    assert all(-2.0 <= c.weight <= 2.0 for c in (*network._input_connections, *network._recurrent_connections, *network._output_connections))


def test_reset_clears_internal_traces_after_plastic_learning():
    network = SparseNetwork(3, 4, 3, density=1.0, seed=7, input_plasticity=0.01, recurrent_plasticity=0.05)
    _train_step(network)
    network.reset()

    assert network._input_eligibility == [0.0] * len(network._input_connections)
    assert network._recurrent_eligibility == [0.0] * len(network._recurrent_connections)
    assert network._eligibility == [0.0] * len(network._output_connections)
