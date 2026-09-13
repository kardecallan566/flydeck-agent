from __future__ import annotations

import random
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


@dataclass(frozen=True, slots=True)
class TrainingResult:
    episodes: int
    average_reward: float
    best_reward: float
    last_reward: float
    successful_episodes: int


class Agent:
    """Minimal agent: observe -> recurrent state -> action -> feedback."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 32,
        density: float = 0.15,
        memory_capacity: int = 256,
        learning_rate: float = 0.02,
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
        self.learning_rate = learning_rate
        self._rng = random.Random(seed + 1)

    def act(self, observation: tuple[float, ...]) -> int:
        scores = self.network.step(observation)
        return max(range(len(scores)), key=scores.__getitem__)

    def act_epsilon_greedy(
        self, observation: tuple[float, ...], epsilon: float
    ) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.network.output_size)
        return self.act(observation)

    def run(self, environment: Environment, max_steps: int = 100) -> AgentResult:
        return self._run_episode(environment, max_steps=max_steps, epsilon=0.0)

    def train(
        self,
        environment: Environment,
        episodes: int = 100,
        max_steps: int = 100,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
    ) -> TrainingResult:
        """Train with simple epsilon-greedy exploration and reward-modulated learning."""
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError("min_epsilon must be between 0 and 1")

        rewards: list[float] = []
        successful = 0
        current_epsilon = epsilon

        for _ in range(episodes):
            result = self._run_episode(
                environment, max_steps=max_steps, epsilon=current_epsilon
            )
            rewards.append(result.total_reward)
            if result.total_reward >= 10.0:
                successful += 1
            current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

        return TrainingResult(
            episodes=episodes,
            average_reward=sum(rewards) / len(rewards),
            best_reward=max(rewards),
            last_reward=rewards[-1],
            successful_episodes=successful,
        )

    def _run_episode(
        self, environment: Environment, max_steps: int, epsilon: float
    ) -> AgentResult:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        observation = environment.reset()
        self.network.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = (
                self.act_epsilon_greedy(observation, epsilon)
                if epsilon > 0.0
                else self.act(observation)
            )
            result = environment.step(action)
            self.memory.add(observation, action, result.reward, result.observation, result.done)
            self.network.learn(action, result.reward, self.learning_rate)
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
