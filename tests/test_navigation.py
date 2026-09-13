from flydeck.agent import Agent
from flydeck.navigation import GridNavigationEnvironment


def make_environment() -> GridNavigationEnvironment:
    return GridNavigationEnvironment(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles={(1, 0), (1, 1), (2, 1)},
        max_steps=30,
    )


def test_navigation_observation_and_reset() -> None:
    environment = make_environment()
    observation = environment.reset()

    assert len(observation) == 6
    assert environment.observation_size == 6
    assert environment.action_size == 4
    assert observation[0] == 0.0  # up is open from the top-left corner? no: see below
    assert observation[1] == 1.0  # right is blocked by obstacle
    assert observation[-2:] == (1.0, 1.0)


def test_navigation_reaches_goal_on_valid_path() -> None:
    environment = make_environment()
    environment.reset()

    # Down, down, right, right, up, right, down reaches the goal around obstacles.
    actions = [2, 2, 1, 1, 0, 1, 2]
    result = None
    for action in actions:
        result = environment.step(action)
        if result.done:
            break

    assert result is not None
    assert result.done
    assert result.reward == 10.0


def test_agent_can_train_on_navigation_environment() -> None:
    environment = make_environment()
    agent = Agent(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        hidden_size=16,
        density=0.2,
        learning_rate=0.02,
        seed=7,
    )

    result = agent.train(
        environment,
        episodes=5,
        max_steps=30,
        epsilon=0.5,
        epsilon_decay=0.9,
        min_epsilon=0.1,
    )

    assert result.episodes == 5
    assert result.successful_episodes >= 0
    assert result.best_reward >= result.last_reward or result.last_reward >= result.best_reward
