from __future__ import annotations

from .agent import Agent
from .environment import CounterEnvironment
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_training import (
    evaluate_buy_and_hold,
    evaluate_hold,
    evaluate_synthetic_crypto,
    train_synthetic_crypto,
)
from .navigation import GridNavigationEnvironment


def main() -> None:
    environment = CryptoTradingEnvironment(SyntheticCryptoMarket(length=256, seed=42).generate(), max_steps=200)
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.01,
        seed=42,
    )

    print("FlyDeck Agent - Synthetic Crypto")
    print(f"observations: {environment.observation_size}")
    print("actions: HOLD / BUY / SELL")
    print(f"connections: {agent.network.connection_count}")
    print()

    training = train_synthetic_crypto(
        agent,
        episodes=100,
        market_length=256,
        max_steps=200,
        seed=100,
        epsilon=0.35,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        trade_penalty=0.0005,
        drawdown_penalty=0.02,
    )

    print(f"episodes: {training.episodes}")
    print(f"average return: {training.average_return_pct:.3f}%")
    print(f"best return: {training.best_return_pct:.3f}%")
    print(f"last return: {training.last_return_pct:.3f}%")
    print(f"average drawdown: {training.average_drawdown_pct:.3f}%")
    print(f"trades: {training.total_trades}")
    print("training actions:")
    print(f"  HOLD: {training.hold_actions}")
    print(f"  BUY:  {training.buy_actions}")
    print(f"  SELL: {training.sell_actions}")

    evaluation_seed = 10_000
    evaluation = evaluate_synthetic_crypto(
        agent,
        seed=evaluation_seed,
        market_length=256,
        max_steps=200,
        trade_penalty=0.0005,
        drawdown_penalty=0.02,
    )
    hold = evaluate_hold(seed=evaluation_seed, market_length=256, max_steps=200)
    buy_hold = evaluate_buy_and_hold(seed=evaluation_seed, market_length=256, max_steps=200)

    print()
    print("Unseen-market evaluation:")
    print(f"agent return: {evaluation.return_pct:.3f}%")
    print(f"agent portfolio: {evaluation.final_portfolio:.2f}")
    print(f"agent drawdown: {evaluation.max_drawdown_pct:.3f}%")
    print(f"agent trades: {evaluation.trades}")
    print("agent actions:")
    print(f"  HOLD: {evaluation.hold_actions}")
    print(f"  BUY:  {evaluation.buy_actions}")
    print(f"  SELL: {evaluation.sell_actions}")
    print()
    print("Baselines on the same unseen market:")
    print(f"  HOLD:      {hold.return_pct:.3f}% | drawdown {hold.max_drawdown_pct:.3f}% | trades {hold.trades}")
    print(f"  BUY&HOLD:  {buy_hold.return_pct:.3f}% | drawdown {buy_hold.max_drawdown_pct:.3f}% | trades {buy_hold.trades}")
    print()
    print(f"connections: {agent.network.connection_count}")
    print(f"memory: {len(agent.memory)}")


def navigation_demo() -> None:
    environment = GridNavigationEnvironment(width=6, height=6, start=(0, 0), goal=(5, 5), max_steps=60)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size)
    result = agent.train(environment, episodes=500, max_steps=60, epsilon=0.35, epsilon_decay=0.995, min_epsilon=0.03)
    print(f"navigation average reward: {result.average_reward:.3f}")


def counter_demo() -> None:
    environment = CounterEnvironment(target=10, max_steps=32)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size)
    result = agent.run(environment)
    print(f"steps: {result.steps}")
    print(f"reward: {result.total_reward:.3f}")


if __name__ == "__main__":
    main()
