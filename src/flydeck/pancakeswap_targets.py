from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset


@dataclass(frozen=True, slots=True)
class PancakeRound:
    epoch: int
    start_timestamp_ms: int
    lock_timestamp_ms: int
    close_timestamp_ms: int
    lock_price: float
    close_price: float
    oracle_called: bool = True
    total_amount: float | None = None
    bull_amount: float | None = None
    bear_amount: float | None = None
    reward_base_cal_amount: float | None = None
    reward_amount: float | None = None

    @property
    def outcome(self) -> Prediction:
        if not self.oracle_called or self.lock_price <= 0.0 or self.close_price <= 0.0:
            return Prediction.WAIT
        if self.close_price > self.lock_price:
            return Prediction.UP
        if self.close_price < self.lock_price:
            return Prediction.DOWN
        return Prediction.WAIT


@dataclass(frozen=True, slots=True)
class AlignedPancakeRound:
    epoch: int
    feature_index: int
    feature_candle_open_ms: int
    decision_timestamp_ms: int
    lock_timestamp_ms: int
    close_timestamp_ms: int
    outcome: Prediction


@dataclass(frozen=True, slots=True)
class PancakeAlignment:
    dataset: BNBPredictionDataset
    aligned_rounds: tuple[AlignedPancakeRound, ...]
    skipped_invalid: int
    skipped_outside_market_history: int
    skipped_duplicate_feature_index: int

    @property
    def eligible_rounds(self) -> int:
        return sum(row.outcome in (Prediction.UP, Prediction.DOWN) for row in self.aligned_rounds)


def load_pancake_rounds_csv(path: str | Path) -> tuple[PancakeRound, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"PancakeSwap rounds CSV is empty: {source}")

    aliases = {_normalize_name(name): name for name in fieldnames}
    required = ("epoch", "starttimestamp", "locktimestamp", "closetimestamp", "lockprice", "closeprice")
    missing = [name for name in required if name not in aliases]
    if missing:
        raise ValueError(f"missing PancakeSwap round columns: {missing}")

    def value(row: dict[str, str], name: str) -> str:
        return str(row[aliases[name]]).strip()

    def optional_float(row: dict[str, str], name: str) -> float | None:
        column = aliases.get(name)
        if column is None:
            return None
        raw = str(row.get(column, "")).strip()
        return None if not raw else float(raw)

    rounds: list[PancakeRound] = []
    for row in rows:
        oracle_column = aliases.get("oraclecalled")
        oracle_called = True if oracle_column is None else _parse_bool(str(row.get(oracle_column, "")))
        rounds.append(
            PancakeRound(
                epoch=int(float(value(row, "epoch"))),
                start_timestamp_ms=_timestamp_ms(value(row, "starttimestamp")),
                lock_timestamp_ms=_timestamp_ms(value(row, "locktimestamp")),
                close_timestamp_ms=_timestamp_ms(value(row, "closetimestamp")),
                lock_price=float(value(row, "lockprice")),
                close_price=float(value(row, "closeprice")),
                oracle_called=oracle_called,
                total_amount=optional_float(row, "totalamount"),
                bull_amount=optional_float(row, "bullamount"),
                bear_amount=optional_float(row, "bearamount"),
                reward_base_cal_amount=optional_float(row, "rewardbasecalamount"),
                reward_amount=optional_float(row, "rewardamount"),
            )
        )
    return tuple(sorted(rounds, key=lambda item: (item.lock_timestamp_ms, item.epoch)))


def align_pancake_rounds_to_market(
    market: BNBPredictionDataset,
    rounds: Iterable[PancakeRound],
    *,
    decision_lead_seconds: int = 30,
    candle_interval_ms: int = 300_000,
) -> PancakeAlignment:
    """Map exact Chainlink round outcomes onto the latest fully closed Binance candle.

    Binance timestamps are candle open times. A feature candle is eligible only
    when its close time is at or before the decision cutoff. This prevents a
    partially formed 5-minute candle from leaking information from the betting
    window into the benchmark.
    """
    if decision_lead_seconds < 0:
        raise ValueError("decision_lead_seconds must be non-negative")
    if candle_interval_ms <= 0:
        raise ValueError("candle_interval_ms must be positive")

    overrides: list[Prediction | None] = [None] * market.size
    aligned: list[AlignedPancakeRound] = []
    skipped_invalid = 0
    skipped_outside = 0
    skipped_duplicate = 0

    for round_row in sorted(rounds, key=lambda item: (item.lock_timestamp_ms, item.epoch)):
        outcome = round_row.outcome
        if outcome not in (Prediction.UP, Prediction.DOWN):
            skipped_invalid += 1
            continue

        decision_timestamp_ms = round_row.lock_timestamp_ms - decision_lead_seconds * 1000
        latest_open_ms = decision_timestamp_ms - candle_interval_ms
        feature_index = bisect_right(market.timestamps, latest_open_ms) - 1
        if feature_index < 0 or feature_index >= market.size - 1:
            skipped_outside += 1
            continue
        if overrides[feature_index] is not None:
            skipped_duplicate += 1
            continue

        overrides[feature_index] = outcome
        aligned.append(
            AlignedPancakeRound(
                epoch=round_row.epoch,
                feature_index=feature_index,
                feature_candle_open_ms=market.timestamps[feature_index],
                decision_timestamp_ms=decision_timestamp_ms,
                lock_timestamp_ms=round_row.lock_timestamp_ms,
                close_timestamp_ms=round_row.close_timestamp_ms,
                outcome=outcome,
            )
        )

    dataset = market.with_outcome_overrides(
        tuple(overrides),
        target_name=f"pancakeswap-lock-close-lead-{decision_lead_seconds}s",
    )
    return PancakeAlignment(
        dataset=dataset,
        aligned_rounds=tuple(aligned),
        skipped_invalid=skipped_invalid,
        skipped_outside_market_history=skipped_outside,
        skipped_duplicate_feature_index=skipped_duplicate,
    )


def _normalize_name(name: str) -> str:
    return "".join(char for char in str(name).strip().lower() if char.isalnum())


def _timestamp_ms(raw: str) -> int:
    value = int(float(raw))
    return value * 1000 if abs(value) < 1_000_000_000_000 else value


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")
