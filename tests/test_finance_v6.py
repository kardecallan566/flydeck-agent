import math

from flydeck.agent import Agent
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_training import evaluate_synthetic_crypto, train_synthetic_crypto


def make_v6_agent(seed: int = 42) -> Agent:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    return Agent(
        observation_size=encoder.output_size,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=seed,
    )


def test_sparse_market_encoder_is_sparse_and_deterministic() -> None:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    observation = tuple((index - 6) / 6 for index in range(12))
    first = encoder.encode(observation)
    encoder.reset()
    second = encoder.encode(observation)

    assert first == second
    assert len(first) == 27
    assert sum(value != 0.0 for value in first) == 5
    assert encoder.active_units == 5
    assert math.isclose(encoder.sparsity, 5 / 27)


def test_sparse_encoder_tracks_previous_action_without_growing_network() -> None:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    observation = tuple(0.1 * index for index in range(12))
    before = encoder.encode(observation)
    encoder.observe_action(2)
    after = encoder.encode(observation)

    assert before[-3:] == (1.0, 0.0, 0.0)
    assert after[-3:] == (0.0, 0.0, 1.0)
    assert len(after) == len(before) == 27


def test_v6_finance_training_is_deterministic() -> None:
    first = make_v6_agent(seed=8)
    second = make_v6_agent(seed=8)
    first_result = train_synthetic_crypto(
        first, episodes=3, market_length=96, max_steps=40, seed=300,
        epsilon=0.4, epsilon_decay=0.9, min_epsilon=0.1,
        opportunity_margin=0.08,
    )
    second_result = train_synthetic_crypto(
        second, episodes=3, market_length=96, max_steps=40, seed=300,
        epsilon=0.4, epsilon_decay=0.9, min_epsilon=0.1,
        opportunity_margin=0.08,
    )

    assert first_result == second_result
    assert first_result.total_actions == 120
    assert first_result.gated_hold_actions >= 0
    assert math.isfinite(first_result.average_return_pct)


def test_v6_evaluation_uses_sparse_state_and_opportunity_gate() -> None:
    agent = make_v6_agent(seed=11)
    result = evaluate_synthetic_crypto(
        agent, seed=9_999, market_length=96, max_steps=40,
        opportunity_margin=0.08,
    )

    assert result.total_actions == 40
    assert result.gated_hold_actions >= 0
    assert result.hold_actions >= result.gated_hold_actions
    assert 0 <= result.trades <= result.buy_actions + result.sell_actions
    assert math.isfinite(result.return_pct)
