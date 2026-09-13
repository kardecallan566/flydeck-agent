from __future__ import annotations

from .agent import Agent
from .environment import CounterEnvironment
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_training import train_synthetic_crypto
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
    )

    print(f"episodes: {training.episodes}")
    print(f"average return: {training.average_return_pct:.3f}%")
    print(f"best return: {training.best_return_pct:.3f}%")
    print(f"last return: {training.last_return_pct:.3f}%")
    print(f"average drawdown: {training.average_drawdown_pct:.3f}%")
    print(f"trades: {training.total_trades}")

    evaluation_environment = CryptoTradingEnvironment(
        SyntheticCryptoMarket(length=256, seed=10_000).generate(),
        max_steps=200,
    )
    evaluation = agent.run(evaluation_environment, max_steps=200)
    metrics = evaluation_environment.episode_result(evaluation.total_reward)
    print()
    print("Unseen-market evaluation:")
    print(f"return: {metrics.return_pct:.3f}%")
    print(f"final portfolio: {metrics.final_portfolio:.2f}")
    print(f"max drawdown: {metrics.max_drawdown_pct:.3f}%")
    print(f"trades: {metrics.trades}")
    print(f"connections: {evaluation.connection_count}")
    print(f"memory: {evaluation.memory_size}")


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
