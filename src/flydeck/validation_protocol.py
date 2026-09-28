from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexRange:
    """Half-open causal range [start, end) of prediction indices."""

    start: int
    end: int

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class ChronologicalProtocol:
    """Frozen chronological train/validation/test ranges with boundary purging."""

    train: IndexRange
    validation: IndexRange
    test: IndexRange
    purge: int
    context: int

    def assert_disjoint(self) -> None:
        if not (self.train.end <= self.validation.start <= self.validation.end <= self.test.start):
            raise AssertionError("chronological ranges overlap or are out of order")


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train: IndexRange
    validation: IndexRange
    test: IndexRange


def build_chronological_protocol(
    dataset_size: int,
    *,
    context: int = 32,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge: int = 1,
) -> ChronologicalProtocol:
    """Build a strict t -> t+1 protocol.

    Prediction index i may use candles <= i and its label is candle i+1.
    A purge gap around split boundaries prevents a training label from sharing
    the first feature candle of the following split.
    """
    if dataset_size < context + 20:
        raise ValueError("dataset is too small for the requested context")
    if context < 2:
        raise ValueError("context must be at least two candles")
    if purge < 1:
        raise ValueError("purge must be at least one prediction index")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave a test split")

    first = context - 1
    usable_end = dataset_size - 1  # last valid prediction index is dataset_size - 2
    span = usable_end - first
    cut_train = first + int(span * train_fraction)
    cut_validation = first + int(span * (train_fraction + validation_fraction))

    protocol = ChronologicalProtocol(
        train=IndexRange(first, cut_train - purge),
        validation=IndexRange(cut_train + purge, cut_validation - purge),
        test=IndexRange(cut_validation + purge, usable_end),
        purge=purge,
        context=context,
    )
    protocol.assert_disjoint()
    if min(protocol.train.size, protocol.validation.size, protocol.test.size) <= 0:
        raise ValueError("purge/context leave an empty split")
    return protocol


def build_walk_forward_folds(
    dataset_size: int,
    *,
    context: int = 32,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    purge: int = 1,
    expanding_train: bool = True,
) -> tuple[WalkForwardFold, ...]:
    """Create causal purged walk-forward folds without shuffling."""
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("fold sizes must be positive")
    if purge < 1:
        raise ValueError("purge must be at least one")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    first = context - 1
    usable_end = dataset_size - 1
    folds: list[WalkForwardFold] = []
    anchor = first + train_size
    while True:
        train_start = first if expanding_train else anchor - train_size
        train_end = anchor - purge
        val_start = anchor + purge
        val_end = val_start + validation_size
        test_start = val_end + purge
        test_end = test_start + test_size
        if test_end > usable_end:
            break
        folds.append(
            WalkForwardFold(
                train=IndexRange(train_start, train_end),
                validation=IndexRange(val_start, val_end),
                test=IndexRange(test_start, test_end),
            )
        )
        anchor += step
    return tuple(folds)
