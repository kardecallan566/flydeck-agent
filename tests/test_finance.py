import math

from flydeck.agent import Agent
from flydeck.finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from flydeck.finance_training import (
    evaluate_buy_and_hold,
    evaluate_hold,
    evaluate_synthetic_crypto,
    train_synthetic_crypto,
)


def make_environment(seed: int = 7, **kwargs) -> CryptoTradingEnvironment:
    market = SyntheticCryptoMarket(length=160, seed=seed).generate()
    return CryptoTradingEnvironment(market, window=24, max_steps=100, **kwargs)


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
    result = environment.step(1)
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


def test_invalid_trade_is_penalized_and_not_counted_as_trade() -> None:
    environment = make_environment(invalid_action_penalty=0.5)
    environment.reset()
    result = environment.step(2)  # sell without a position
    assert environment.trades == 0
    assert environment.invalid_actions == 1
    assert result.reward < 0.0


def test_risk_aware_reward_penalizes_trade_and_drawdown() -> None:
    market = SyntheticCryptoMarket(length=96, seed=77).generate()
    plain = CryptoTradingEnvironment(market, window=24, max_steps=40, trade_penalty=0.0, drawdown_penalty=0.0)
    risk = CryptoTradingEnvironment(market, window=24, max_steps=40, trade_penalty=0.01, drawdown_penalty=0.10)
    plain.reset()
    risk.reset()
    plain_result = plain.step(1)
    risk_result = risk.step(1)
    assert risk_result.reward <= plain_result.reward


def test_agent_can_train_on_unpredictable_market() -> None:
    environment = make_environment(seed=21)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size,
                  hidden_size=32, density=0.10, learning_rate=0.01, seed=21)
    result = agent.train(environment, episodes=5, max_steps=80, epsilon=0.5, epsilon_decay=0.9, min_epsilon=0.1)
    assert result.episodes == 5
    assert result.best_reward >= result.last_reward
    assert math.isfinite(result.average_reward)
    assert math.isfinite(result.best_reward)


def test_finance_training_reports_action_distribution() -> None:
    environment = make_environment(seed=31)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size,
                  hidden_size=16, density=0.10, learning_rate=0.01, seed=31)
    result = train_synthetic_crypto(agent, episodes=3, market_length=96, max_steps=40, seed=50,
                                    epsilon=0.5, epsilon_decay=0.9, min_epsilon=0.1)
    assert result.total_actions == result.episodes * 40
    assert 0 <= result.total_trades <= result.buy_actions + result.sell_actions
    assert 0.0 <= result.trade_frequency <= 1.0
    assert math.isfinite(result.average_return_pct)
    assert math.isfinite(result.average_drawdown_pct)


def test_greedy_evaluation_returns_action_distribution() -> None:
    environment = make_environment(seed=41)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size,
                  hidden_size=16, density=0.10, learning_rate=0.01, seed=41)
    result = evaluate_synthetic_crypto(agent, seed=9_999, market_length=96, max_steps=40)
    assert result.total_actions == 40
    assert 0 <= result.trades <= result.buy_actions + result.sell_actions
    assert 0.0 <= result.trade_frequency <= 1.0
    assert math.isfinite(result.return_pct)
    assert math.isfinite(result.max_drawdown_pct)


def test_baselines_are_deterministic_and_buy_hold_trades_once() -> None:
    hold = evaluate_hold(seed=123, market_length=96, max_steps=40)
    hold_again = evaluate_hold(seed=123, market_length=96, max_steps=40)
    buy_hold = evaluate_buy_and_hold(seed=123, market_length=96, max_steps=40)
    assert hold == hold_again
    assert hold.trades == 0
    assert buy_hold.trades == 1
    assert math.isfinite(hold.return_pct)
    assert math.isfinite(buy_hold.return_pct)
