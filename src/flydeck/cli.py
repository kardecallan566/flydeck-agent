from __future__ import annotations

from .agent import Agent
from .environment import CounterEnvironment


def main() -> None:
    environment = CounterEnvironment(target=10, max_steps=32)
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.15,
    )
    result = agent.run(environment)

    print("FlyDeck Agent")
    print(f"steps: {result.steps}")
    print(f"reward: {result.total_reward:.3f}")
    print(f"connections: {result.connection_count}")
    print(f"memory: {result.memory_size}")


if __name__ == "__main__":
    main()
