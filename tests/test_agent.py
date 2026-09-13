from flydeck.agent import Agent
from flydeck.environment import CounterEnvironment


def test_agent_runs_and_records_experience():
    environment = CounterEnvironment(target=3, max_steps=10)
    agent = Agent(1, 2, hidden_size=8, density=0.5, memory_capacity=10, seed=3)

    result = agent.run(environment)

    assert result.steps > 0
    assert result.steps <= 10
    assert result.memory_size == result.steps
    assert result.connection_count > 0


def test_memory_is_bounded():
    environment = CounterEnvironment(target=100, max_steps=20)
    agent = Agent(1, 2, hidden_size=8, memory_capacity=5, seed=3)

    result = agent.run(environment, max_steps=20)

    assert result.memory_size == 5
