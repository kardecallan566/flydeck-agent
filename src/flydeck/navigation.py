from __future__ import annotations

from dataclasses import dataclass

from .environment import StepResult


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


class GridNavigationEnvironment:
    """Small deterministic navigation task for validating the agent runtime."""

    # Actions: up, right, down, left.
    ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        obstacles: set[tuple[int, int]] | None = None,
        max_steps: int = 60,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("grid dimensions must be >= 2")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.width = width
        self.height = height
        self.start = Position(*start)
        self.goal = Position(*(goal if goal is not None else (width - 1, height - 1)))
        self.obstacles = {Position(x, y) for x, y in (obstacles or set())}
        self.max_steps = max_steps

        for position in (self.start, self.goal):
            if not self._inside(position):
                raise ValueError("start and goal must be inside the grid")
            if position in self.obstacles:
                raise ValueError("start and goal cannot be obstacles")

        self.position = self.start
        self.steps = 0

    @property
    def observation_size(self) -> int:
        # Four blocked-neighbor flags + normalized direction to goal.
        return 6

    @property
    def action_size(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> tuple[float, ...]:
        self.position = self.start
        self.steps = 0
        return self._observation()

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.action_size:
            raise ValueError("action is outside the environment action range")

        dx, dy = self.ACTIONS[action]
        candidate = Position(self.position.x + dx, self.position.y + dy)
        blocked = not self._inside(candidate) or candidate in self.obstacles

        previous_distance = self._distance(self.position, self.goal)
        if not blocked:
            self.position = candidate
        current_distance = self._distance(self.position, self.goal)

        self.steps += 1
        if self.position == self.goal:
            return StepResult(self._observation(), 10.0, True)
        if self.steps >= self.max_steps:
            return StepResult(self._observation(), -1.0, True)

        reward = -0.05
        if blocked:
            reward -= 0.5
        elif current_distance < previous_distance:
            reward += 0.15
        elif current_distance > previous_distance:
            reward -= 0.15

        return StepResult(self._observation(), reward, False)

    def render(self) -> str:
        rows: list[str] = []
        for y in range(self.height):
            row: list[str] = []
            for x in range(self.width):
                position = Position(x, y)
                if position == self.position:
                    row.append("A")
                elif position == self.goal:
                    row.append("G")
                elif position in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def _observation(self) -> tuple[float, ...]:
        blocked = []
        for dx, dy in self.ACTIONS:
            candidate = Position(self.position.x + dx, self.position.y + dy)
            blocked.append(1.0 if not self._inside(candidate) or candidate in self.obstacles else 0.0)

        dx = (self.goal.x - self.position.x) / max(1, self.width - 1)
        dy = (self.goal.y - self.position.y) / max(1, self.height - 1)
        return (*blocked, dx, dy)

    def _inside(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    @staticmethod
    def _distance(first: Position, second: Position) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)
