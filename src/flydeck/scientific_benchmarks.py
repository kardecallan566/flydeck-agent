from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset


class BaselinePredictor(Protocol):
    """Protocol for simple market baselines."""

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction: ...
    def reset(self) -> None: ...


class PreviousDirectionBaseline:
    """Lag-1 persistence baseline: predicts continuation of previous candle."""

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        if index < 1:
            return Prediction.WAIT
        curr = dataset.closes[index]
        prev = dataset.closes[index - 1]
        if curr > prev:
            return Prediction.UP
        if curr < prev:
            return Prediction.DOWN
        return Prediction.WAIT

    def reset(self) -> None:
        pass


class MomentumBaseline:
    """Rolling momentum baseline: predicts direction based on return over window."""

    def __init__(self, window: int = 6) -> None:
        if window < 1:
            raise ValueError("window must be at least one")
        self.window = window

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        if index < self.window:
            return Prediction.WAIT
        curr = dataset.closes[index]
        past = dataset.closes[index - self.window]
        if curr > past:
            return Prediction.UP
        if curr < past:
            return Prediction.DOWN
        return Prediction.WAIT

    def reset(self) -> None:
        pass


class AlwaysUpBaseline:
    """Always predicts UP."""

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        return Prediction.UP

    def reset(self) -> None:
        pass


class AlwaysDownBaseline:
    """Always predicts DOWN."""

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        return Prediction.DOWN

    def reset(self) -> None:
        pass


class RandomBaseline:
    """Random coin-flip baseline."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def predict(self, dataset: BNBPredictionDataset, index: int) -> Prediction:
        return self._rng.choice((Prediction.UP, Prediction.DOWN))

    def reset(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Detailed evaluation metrics for binary directional prediction with abstention (WAIT)."""

    name: str
    total_rounds: int
    true_up: int
    false_up: int
    true_down: int
    false_down: int
    waits: int

    @property
    def entered_rounds(self) -> int:
        return self.true_up + self.false_up + self.true_down + self.false_down

    @property
    def correct(self) -> int:
        return self.true_up + self.true_down

    @property
    def accuracy(self) -> float:
        return self.correct / self.entered_rounds if self.entered_rounds else 0.0

    @property
    def coverage(self) -> float:
        return self.entered_rounds / self.total_rounds if self.total_rounds else 0.0

    @property
    def precision_up(self) -> float:
        total = self.true_up + self.false_up
        return self.true_up / total if total else 0.0

    @property
    def precision_down(self) -> float:
        total = self.true_down + self.false_down
        return self.true_down / total if total else 0.0

    @property
    def f1_score(self) -> float:
        p_up = self.precision_up
        p_down = self.precision_down
        if p_up + p_down <= 1e-12:
            return 0.0
        return 2.0 * (p_up * p_down) / (p_up + p_down)

    def pancakeswap_expectancy(self, treasury_fee: float = 0.03) -> float:
        """Net expected multiplier per bet on PancakeSwap Prediction (1.0 = breakeven).

        With 3% treasury fee, win payout is 1.0 * (1 - fee) = +0.97, loss is -1.00.
        Net edge = accuracy * (1 - fee) - (1 - accuracy) * 1.0.
        """
        if self.entered_rounds == 0:
            return 0.0
        acc = self.accuracy
        return acc * (1.0 - treasury_fee) - (1.0 - acc)


def evaluate_predictions(
    name: str,
    predictions: tuple[Prediction, ...],
    outcomes: tuple[Prediction, ...],
) -> ConfusionMatrix:
    """Compute complete confusion matrix from a sequence of decisions and outcomes."""
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have the same length")

    true_up = false_up = true_down = false_down = waits = 0
    for pred, outcome in zip(predictions, outcomes):
        if pred == Prediction.WAIT or outcome == Prediction.WAIT:
            waits += 1
        elif pred == Prediction.UP:
            if outcome == Prediction.UP:
                true_up += 1
            else:
                false_up += 1
        elif pred == Prediction.DOWN:
            if outcome == Prediction.DOWN:
                true_down += 1
            else:
                false_down += 1

    return ConfusionMatrix(
        name=name,
        total_rounds=len(predictions),
        true_up=true_up,
        false_up=false_up,
        true_down=true_down,
        false_down=false_down,
        waits=waits,
    )


def evaluate_baseline(
    baseline: BaselinePredictor,
    dataset: BNBPredictionDataset,
    start: int,
    end: int,
    name: str,
) -> ConfusionMatrix:
    """Run a baseline predictor over a dataset range and compute metrics."""
    baseline.reset()
    predictions = tuple(baseline.predict(dataset, index) for index in range(start, end))
    outcomes = tuple(dataset.outcome(index) for index in range(start, end))
    return evaluate_predictions(name, predictions, outcomes)


def evaluate_by_regimes(
    name: str,
    predictions: tuple[Prediction, ...],
    dataset: BNBPredictionDataset,
    start: int,
    end: int,
) -> dict[str, ConfusionMatrix]:
    """Segment model performance across market regimes (high/low vol, trending/chop)."""
    outcomes = tuple(dataset.outcome(index) for index in range(start, end))
    indices = range(start, end)

    results = {
        "overall": evaluate_predictions(f"{name}_overall", predictions, outcomes),
    }

    # Volatility segments
    high_vol = [
        (p, o)
        for p, o, idx in zip(predictions, outcomes, indices)
        if dataset.volatility_regime(idx) == "high"
    ]
    low_vol = [
        (p, o)
        for p, o, idx in zip(predictions, outcomes, indices)
        if dataset.volatility_regime(idx) == "low"
    ]
    if high_vol:
        preds, outs = zip(*high_vol)
        results["volatility_high"] = evaluate_predictions(f"{name}_high_vol", preds, outs)
    if low_vol:
        preds, outs = zip(*low_vol)
        results["volatility_low"] = evaluate_predictions(f"{name}_low_vol", preds, outs)

    # Trend segments
    trending = [
        (p, o)
        for p, o, idx in zip(predictions, outcomes, indices)
        if dataset.trend_regime(idx) == "trending"
    ]
    chop = [
        (p, o)
        for p, o, idx in zip(predictions, outcomes, indices)
        if dataset.trend_regime(idx) == "chop"
    ]
    if trending:
        preds, outs = zip(*trending)
        results["trend_trending"] = evaluate_predictions(f"{name}_trending", preds, outs)
    if chop:
        preds, outs = zip(*chop)
        results["trend_chop"] = evaluate_predictions(f"{name}_chop", preds, outs)

    return results
