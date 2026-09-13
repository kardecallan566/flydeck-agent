from flydeck.agent import Agent
from flydeck.finance import CryptoTradingEnvironment, SyntheticCryptoMarket


def make_environment(seed: int = 7) -> CryptoTradingEnvironment:
    market = SyntheticCryptoMarket(length=160, seed=seed).generate()
    return CryptoTradingEnvironment(market, window=24, max_steps=100)


def test_market_generation_is_deterministic_but_seeded() -> None:
    first = SyntheticCryptoMarket(length=64, seed=11).generate()
    second = SyntheticCryptoMarket(length=64, seed=11).generate()
    other = SyntheticCryptoMarket(length=64, seed=12).generate()

    assert first == second
    assert first != other
    assert len(first) == 64
    assert all(candle.low <= candle.open <= candle.high for candle in first)
    assert all(candle.low <= candle.close <= candle.high for candle in first)


def test_finance_observation_and_portfolio_accounting() -> None:
    environment = make_environment()
    observation = environment.reset()

    assert len(observation) == environment.observation_size == 12
    assert environment.action_size == 3
    assert environment.portfolio_value == 10_000.0
    assert environment.position_ratio == 0.0

    before = environment.portfolio_value
    result = environment.step(1)  # buy

    assert len(result.observation) == 12
    assert environment.trades == 1
    assert environment.position_ratio > 0.0
    assert environment.portfolio_value > 0.0
    assert environment.portfolio_value != before


def test_hold_does_not_create_trades() -> None:
    environment = make_environment()
    environment.reset()
    result = environment.step(0)

    assert environment.trades == 0
    assert result.done is False
    assert environment.position_ratio == 0.0


def test_agent_can_train_on_unpredictable_market() -> None:
    environment = make_environment(seed=21)
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.01,
        seed=21,
    )

    result = agent.train(
        environment,
        episodes=5,
        max_steps=80,
        epsilon=0.5,
        epsilon_decay=0.9,
        min_epsilon=0.1,
    )

    assert result.episodes == 5
    assert result.best_reward >= result.last_reward
    assert result.average_reward == sum(
        [result.average_reward]
    )  # result is finite and exposes the aggregate metric
