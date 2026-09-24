from __future__ import annotations

from dataclasses import dataclass

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
) -> tuple[FlyVisualPredictionAgent, SurvivalTrainingResult]:
    """Train causally with finite lives on a chronological market segment.

    The action at candle ``t`` sees only candles through ``t``. Its life is
    reduced only after candle ``t+1`` is available and the direction is known.
    WAIT is not counted as a prediction and does not lose a life; this avoids
    teaching the agent to gamble merely to stay alive. On death, neural state
    is reset while learned MBON associations are retained by default.
    """
    if context < 4:
        raise ValueError("context must be at least four candles")
    if initial_lives < 1:
        raise ValueError("initial_lives must be at least one")
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
    deaths = correct = entered = up = down = wait = 0

    for index in range(start, end):
        prices = data.closes[max(0, index - context + 1) : index + 1]
        _stimulus, decision = agent.perceive(prices, volumes=data.volumes[max(0, index - context + 1) : index + 1])
        outcome = data.outcome(index)

        if decision.action == Prediction.UP:
            up += 1
            entered += 1
            is_correct = outcome == Prediction.UP
        elif decision.action == Prediction.DOWN:
            down += 1
            entered += 1
            is_correct = outcome == Prediction.DOWN
        else:
            wait += 1
            is_correct = False

        if decision.action != Prediction.WAIT:
            correct += int(is_correct)
            if not is_correct:
                lives -= 1
                if lives == 0:
                    deaths += 1
                    if index + 1 < end:
                        lives = initial_lives
                        agent.reset(preserve_learning=preserve_learning_on_death)

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
