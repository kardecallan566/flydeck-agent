from __future__ import annotations

from .agent import Agent
from .environment import CounterEnvironment
from .navigation import GridNavigationEnvironment


def main() -> None:
    environment = GridNavigationEnvironment(
        width=6,
        height=6,
        start=(0, 0),
        goal=(5, 5),
        obstacles={
            (1, 0), (1, 1), (3, 1), (3, 2),
            (1, 3), (2, 3), (4, 3), (4, 4),
        },
        max_steps=60,
    )
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.15,
        learning_rate=0.02,
    )

    print("FlyDeck Agent - Navigation")
    print("Map:")
    print(environment.render())
    print()
    print(f"connections: {agent.network.connection_count}")

    training = agent.train(
        environment,
        episodes=500,
        max_steps=60,
        epsilon=0.35,
        epsilon_decay=0.995,
        min_epsilon=0.03,
    )

    print(f"episodes: {training.episodes}")
    print(f"average reward: {training.average_reward:.3f}")
    print(f"best reward: {training.best_reward:.3f}")
    print(f"last reward: {training.last_reward:.3f}")
    print(f"successful episodes: {training.successful_episodes}/{training.episodes}")

    evaluation = agent.run(environment, max_steps=60)
    print()
    print("Evaluation:")
    print(f"steps: {evaluation.steps}")
    print(f"reward: {evaluation.total_reward:.3f}")
    print(f"memory: {evaluation.memory_size}")


def counter_demo() -> None:
    environment = CounterEnvironment(target=10, max_steps=32)
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
    )
    result = agent.run(environment)
    print(f"steps: {result.steps}")
    print(f"reward: {result.total_reward:.3f}")


if __name__ == "__main__":
    main()
