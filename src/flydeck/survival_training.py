from __future__ import annotations

from dataclasses import dataclass
import random

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class SurvivalTrainingResult:
    rounds: int
    lives_started: int
    deaths: int
    lives_remaining: int
    correct: int
    entered: int
    up: int
    down: int
    wait: int
    exploratory: int
    exploratory_up: int
    exploratory_down: int
    survival_rate: float


def train_visual_survival(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    *,
    context: int = 32,
    initial_lives: int = 3,
    max_rounds: int | None = None,
    confidence_threshold: float = 0.15,
    preserve_learning_on_death: bool = True,
    exploration_rate: float = 0.30,
    min_exploration_rate: float = 0.05,
    exploration_decay: float = 0.995,
    max_wait_streak: int = 8,
    seed: int = 123,
) -> tuple[FlyVisualPredictionAgent, SurvivalTrainingResult]:
    """Train causally with finite lives on a chronological market segment.

    The action at candle ``t`` sees only candles through ``t``. Its life is
    reduced only after candle ``t+1`` is available and the direction is known.
    WAIT is not counted as a prediction and does not lose a life. To prevent
    the degenerate all-WAIT policy, training uses decaying epsilon exploration
    and forces a directional probe after ``max_wait_streak`` waits. Exploratory
    actions are explicitly counted and are not confused with raw decisions.
    On death, neural state is reset while learned MBON associations are kept.
    """
    if context < 4:
        raise ValueError("context must be at least four candles")
    if initial_lives < 1:
        raise ValueError("initial_lives must be at least one")
    if not 0.0 <= min_exploration_rate <= exploration_rate <= 1.0:
        raise ValueError("exploration rates must satisfy 0 <= minimum <= initial <= 1")
    if not 0.0 < exploration_decay <= 1.0:
        raise ValueError("exploration_decay must be in (0, 1]")
    if max_wait_streak < 1:
        raise ValueError("max_wait_streak must be at least one")
    usable = data.size - 1
    start = context - 1
    end = usable if max_rounds is None else min(usable, start + max_rounds)
    if end <= start:
        raise ValueError("dataset is too small for the requested context")

    agent = FlyVisualPredictionAgent(
        circuit, retina_width=context, confidence_threshold=confidence_threshold
    )
    lives = initial_lives
    lives_started = initial_lives
    deaths = correct = entered = up = down = wait = exploratory = 0
    exploratory_up = exploratory_down = 0
    wait_streak = 0
    current_exploration = exploration_rate
    rng = random.Random(seed)

    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1) : index + 1]
        _stimulus, decision = agent.perceive(
            prices,
            volumes=data.volumes[max(0, index - context + 1) : index + 1],
        )
        outcome = data.outcome(index)
        action = decision.action
        if action == Prediction.WAIT:
            wait_streak += 1
            should_explore = (
                rng.random() < current_exploration
                or wait_streak >= max_wait_streak
            )
            if should_explore:
                # Exploration must be action-symmetric. Choosing the current
                # model's larger score is exploitation disguised as
                # exploration and caused the earlier DOWN skew.
                action = Prediction.UP if rng.random() < 0.5 else Prediction.DOWN
                agent.commit_action(action)
                exploratory += 1
                exploratory_up += int(action == Prediction.UP)
                exploratory_down += int(action == Prediction.DOWN)
                wait_streak = 0
        else:
            wait_streak = 0

        if action == Prediction.UP:
            up += 1
            entered += 1
            is_correct = outcome == Prediction.UP
        elif action == Prediction.DOWN:
            down += 1
            entered += 1
            is_correct = outcome == Prediction.DOWN
        else:
            wait += 1
            is_correct = False

        if action != Prediction.WAIT:
            correct += int(is_correct)
            if not is_correct:
                lives -= 1
                if lives == 0:
                    deaths += 1
                    if index + 1 < end:
                        lives = initial_lives
                        agent.reset(preserve_learning=preserve_learning_on_death)
                        wait_streak = 0
        current_exploration = max(
            min_exploration_rate, current_exploration * exploration_decay
        )

    rounds = end - start
    return agent, SurvivalTrainingResult(
        rounds=rounds,
        lives_started=lives_started,
        deaths=deaths,
        lives_remaining=lives,
        correct=correct,
        entered=entered,
        up=up,
        down=down,
        wait=wait,
        exploratory=exploratory,
        exploratory_up=exploratory_up,
        exploratory_down=exploratory_down,
        survival_rate=(rounds - deaths) / rounds,
    )


def run_survival_benchmark(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    *,
    context: int = 32,
    initial_lives: int = 3,
    confidence_threshold: float = 0.15,
) -> SurvivalTrainingResult:
    """Convenience wrapper that discards the trained agent."""
    _agent, result = train_visual_survival(
        data,
        circuit,
        context=context,
        initial_lives=initial_lives,
        confidence_threshold=confidence_threshold,
    )
    return result
