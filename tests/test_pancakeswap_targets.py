from __future__ import annotations

from flydeck.bnb_prediction import Prediction
from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.pancakeswap_targets import (
    PancakeRound,
    align_pancake_rounds_to_market,
    load_pancake_rounds_csv,
)


def _market(n: int = 20) -> BNBPredictionDataset:
    closes = tuple(100.0 + i for i in range(n))
    return BNBPredictionDataset(
        timestamps=tuple(i * 300_000 for i in range(n)),
        opens=closes,
        highs=tuple(value + 1.0 for value in closes),
        lows=tuple(value - 1.0 for value in closes),
        closes=closes,
        volumes=tuple(10.0 for _ in range(n)),
    )


def test_alignment_uses_only_fully_closed_candle_before_decision_cutoff() -> None:
    market = _market()
    round_row = PancakeRound(
        epoch=123,
        start_timestamp_ms=1_200_000,
        lock_timestamp_ms=1_800_000,
        close_timestamp_ms=2_100_000,
        lock_price=100.0,
        close_price=101.0,
        oracle_called=True,
    )
    alignment = align_pancake_rounds_to_market(
        market,
        (round_row,),
        decision_lead_seconds=30,
    )
    assert alignment.eligible_rounds == 1
    aligned = alignment.aligned_rounds[0]
    assert aligned.feature_index == 4
    assert market.timestamps[aligned.feature_index] + 300_000 <= aligned.decision_timestamp_ms
    assert alignment.dataset.outcome(aligned.feature_index) == Prediction.UP
    assert alignment.dataset.target_name == "pancakeswap-lock-close-lead-30s"


def test_cancelled_and_tied_rounds_are_not_binary_targets() -> None:
    market = _market()
    rounds = (
        PancakeRound(1, 600_000, 1_500_000, 1_800_000, 100.0, 100.0, True),
        PancakeRound(2, 900_000, 1_800_000, 2_100_000, 100.0, 101.0, False),
    )
    alignment = align_pancake_rounds_to_market(market, rounds, decision_lead_seconds=0)
    assert alignment.eligible_rounds == 0
    assert alignment.skipped_invalid == 2


def test_round_csv_accepts_contract_style_camel_case_columns(tmp_path) -> None:
    path = tmp_path / "rounds.csv"
    path.write_text(
        "epoch,startTimestamp,lockTimestamp,closeTimestamp,lockPrice,closePrice,oracleCalled\n"
        "42,1000,1300,1600,10000000000,10010000000,true\n",
        encoding="utf-8",
    )
    rows = load_pancake_rounds_csv(path)
    assert len(rows) == 1
    assert rows[0].epoch == 42
    assert rows[0].lock_timestamp_ms == 1_300_000
    assert rows[0].outcome == Prediction.UP
