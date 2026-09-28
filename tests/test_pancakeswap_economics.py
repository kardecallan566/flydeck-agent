from flydeck.scientific_benchmarks import ConfusionMatrix


def test_pancakeswap_scenario_ev_requires_explicit_pool_assumption() -> None:
    matrix = ConfusionMatrix(
        name="scenario",
        total_rounds=100,
        true_up=30,
        false_up=20,
        true_down=30,
        false_down=20,
        waits=0,
    )
    # 60% accuracy. With a balanced 2x gross payout and 3% fee,
    # win profit is +0.94 and loss is -1.00.
    expected = 0.60 * 0.94 - 0.40
    assert abs(matrix.pancakeswap_expectancy(payout_ratio=2.0) - expected) < 1e-12

    # A lower payout can make the same classifier negative-EV.
    assert matrix.pancakeswap_expectancy(payout_ratio=1.5) < 0.0
