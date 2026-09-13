from pathlib import Path

from flydeck.agent import Agent
from flydeck.finance import SyntheticCryptoMarket
from flydeck.finance_data import RealMarketDataset, load_ohlcv_csv, save_ohlcv_csv
from flydeck.finance_encoder import SparseMarketEncoder
from flydeck.finance_real import collect_decision_quality, evaluate_real_market, future_returns, split_real_market, train_real_market_v8


def _dataset(length: int = 96) -> RealMarketDataset:
    candles = SyntheticCryptoMarket(length=length, seed=123).generate()
    timestamps = tuple(1_700_000_000_000 + index * 3_600_000 for index in range(length))
    return RealMarketDataset("TESTUSDT", "1h", candles, timestamps, "test")


def _agent(seed: int = 42) -> Agent:
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    return Agent(observation_size=encoder.output_size, action_size=3, hidden_size=32, density=0.10, learning_rate=0.005, seed=seed)


def test_real_dataset_round_trip(tmp_path: Path):
    source = _dataset()
    path = tmp_path / "market.csv"
    save_ohlcv_csv(source, path)
    loaded = load_ohlcv_csv(path, symbol="TESTUSDT", interval="1h")
    assert loaded.rows == source.rows
    assert loaded.timestamps_ms == source.timestamps_ms
    assert loaded.candles == source.candles


def test_real_market_split_is_chronological_with_context():
    dataset = _dataset(100)
    splits = split_real_market(dataset, train_ratio=0.70, validation_ratio=0.15, context=24)
    assert splits.train.candles[-1] == dataset.candles[69]
    assert splits.validation.candles[0] == dataset.candles[46]
    assert splits.validation.candles[-1] == dataset.candles[84]
    assert splits.test.candles[0] == dataset.candles[61]
    assert splits.test.candles[-1] == dataset.candles[-1]


def test_future_returns_uses_close_to_close_values():
    dataset = _dataset(40)
    result = dict(future_returns(dataset.candles, 24, (1, 3)))
    assert result[1] == dataset.candles[25].close / dataset.candles[24].close - 1.0
    assert result[3] == dataset.candles[27].close / dataset.candles[24].close - 1.0


def test_real_training_and_unseen_evaluation_run():
    dataset = _dataset(128)
    agent = _agent()
    training = train_real_market_v8(agent, dataset.candles, max_steps=60, epsilon=0.10, epsilon_decay=0.99)
    evaluation = evaluate_real_market(agent, dataset.candles, max_steps=40)
    assert training.steps == 60
    assert evaluation.steps == 40
    assert evaluation.trades >= 0
    assert sum(evaluation.action_counts) == 40
    assert len(evaluation.score_means) == 3


def test_decision_quality_records_forward_returns_and_action_counts():
    dataset = _dataset(96)
    quality = collect_decision_quality(_agent(), dataset.candles, max_steps=40)
    assert len(quality.records) == 40
    assert sum(quality.action_counts) == 40
    assert len(quality.average_future_returns_by_action) == 3
    assert all(len(values) == 5 for values in quality.average_future_returns_by_action)
    assert quality.max_buy_streak >= 0
