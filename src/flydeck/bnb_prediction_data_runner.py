from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from .bnb_prediction import Prediction
from .bnb_prediction_agent import BNBObservation, FlyBNBPredictionAgent, FlyDecision
from .malecns import MaleCNSCircuit

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


@dataclass(frozen=True, slots=True)
class BNBPredictionDataset:
    timestamps: tuple[int, ...]
    opens: tuple[float, ...]
    highs: tuple[float, ...]
    lows: tuple[float, ...]
    closes: tuple[float, ...]
    volumes: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.closes)

    def observation(self, index: int, context: int = 24) -> BNBObservation:
        start = max(0, index - context + 1)
        return BNBObservation(
            timestamp=self.timestamps[index],
            prices=self.closes[start:index + 1],
            volumes=self.volumes[start:index + 1],
        )

    def outcome(self, index: int) -> Prediction:
        if self.closes[index + 1] > self.closes[index]:
            return Prediction.UP
        if self.closes[index + 1] < self.closes[index]:
            return Prediction.DOWN
        return Prediction.WAIT


def load_bnb_5m_csv(path: str | Path) -> BNBPredictionDataset:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("BNB CSV is empty")
    names = {name.strip().lower(): name for name in rows[0]}
    required = {"open", "high", "low", "close", "volume"}
    missing = required - names.keys()
    timestamp_name = names.get("timestamp") or names.get("open_time")
    if timestamp_name is None:
        missing.add("timestamp")
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    columns = {key: [] for key in ("timestamps", "opens", "highs", "lows", "closes", "volumes")}
    for row in rows:
        columns["timestamps"].append(int(float(row[timestamp_name])))
        for key in ("opens", "highs", "lows", "closes", "volumes"):
            columns[key].append(float(row[names[key]]))
    timestamps = columns["timestamps"]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise ValueError("BNB data must be strictly chronological")
    return BNBPredictionDataset(**{key: tuple(value) for key, value in columns.items()})


def download_bnb_5m_csv(path: str | Path, limit: int = 1000) -> Path:
    if not 2 <= limit <= 1000:
        raise ValueError("limit must be between 2 and 1000")
    from urllib.parse import urlencode
    query = urlencode({"symbol": "BNBUSDT", "interval": "5m", "limit": limit})
    with urlopen(f"{BINANCE_KLINES_URL}?{query}", timeout=30) as response:
        import json
        rows = json.load(response)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow(row[:6])
    return path


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    rounds: int
    entered: int
    correct: int
    accuracy: float
    coverage: float
    up: int
    down: int
    wait: int


def _run_split(agent: FlyBNBPredictionAgent, data: BNBPredictionDataset, start: int, end: int, learn: bool) -> PredictionMetrics:
    entered = correct = up = down = wait = 0
    for index in range(start, end):
        prediction = agent.predict(data.observation(index))
        outcome = data.outcome(index)
        if prediction.decision == FlyDecision.UP:
            up += 1
            entered += 1
            correct += int(outcome == Prediction.UP)
        elif prediction.decision == FlyDecision.DOWN:
            down += 1
            entered += 1
            correct += int(outcome == Prediction.DOWN)
        else:
            wait += 1
        if learn:
            agent.learn(outcome)
    rounds = end - start
    return PredictionMetrics(
        rounds=rounds,
        entered=entered,
        correct=correct,
        accuracy=correct / entered if entered else 0.0,
        coverage=entered / rounds if rounds else 0.0,
        up=up,
        down=down,
        wait=wait,
    )


def run_bnb_prediction_benchmark(
    data: BNBPredictionDataset,
    circuit: MaleCNSCircuit,
    seed: int = 123,
    confidence_threshold: float = 0.20,
) -> tuple[PredictionMetrics, PredictionMetrics, PredictionMetrics]:
    usable = data.size - 1
    if usable < 30:
        raise ValueError("dataset needs at least 31 candles")
    train_end = int(usable * 0.70)
    validation_end = train_end + int(usable * 0.15)
    agent = FlyBNBPredictionAgent(circuit, seed=seed, confidence_threshold=confidence_threshold)
    train = _run_split(agent, data, 24, train_end, True)
    validation = _run_split(agent, data, train_end, validation_end, False)
    test = _run_split(agent, data, validation_end, usable, False)
    return train, validation, test
