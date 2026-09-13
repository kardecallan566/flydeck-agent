from __future__ import annotations

from dataclasses import dataclass

from .environment import Environment
from .memory import Memory
from .network import SparseNetwork


@dataclass(frozen=True, slots=True)
class AgentResult:
    total_reward: float
    steps: int
    memory_size: int
    connection_count: int


class Agent:
    """Minimal agent: observe -> recurrent state -> action -> feedback."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 32,
        density: float = 0.15,
        memory_capacity: int = 256,
        seed: int = 42,
    ) -> None:
        self.network = SparseNetwork(
            observation_size,
            hidden_size,
            action_size,
            density=density,
            seed=seed,
        )
        self.memory = Memory(memory_capacity)

    def act(self, observation: tuple[float, ...]) -> int:
        scores = self.network.step(observation)
        return max(range(len(scores)), key=scores.__getitem__)

    def run(self, environment: Environment, max_steps: int = 100) -> AgentResult:
        observation = environment.reset()
        self.network.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = self.act(observation)
            result = environment.step(action)
            self.memory.add(observation, action, result.reward, result.observation, result.done)
            total_reward += result.reward
            steps += 1
            observation = result.observation
            if result.done:
                break

        return AgentResult(
            total_reward=total_reward,
            steps=steps,
            memory_size=len(self.memory),
            connection_count=self.network.connection_count,
        )
