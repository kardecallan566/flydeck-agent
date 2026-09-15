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
        if not 0 <= index < self.size - 1:
            raise IndexError("outcome requires an index with a following candle")
        if self.closes[index + 1] > self.closes[index]:
            return Prediction.UP
        if self.closes[index + 1] < self.closes[index]:
            return Prediction.DOWN
        return Prediction.WAIT

    def volatility(self, index: int, window: int = 24) -> float:
        """Normalized average true range over the preceding window."""
        start = max(0, index - window + 1)
        sub_highs = self.highs[start:index + 1]
        sub_lows = self.lows[start:index + 1]
        sub_closes = self.closes[start:index + 1]
        if not sub_closes:
            return 0.0
        ranges = [(h - l) / max(1e-9, c) for h, l, c in zip(sub_highs, sub_lows, sub_closes)]
        return sum(ranges) / len(ranges)

    def volatility_regime(self, index: int, window: int = 24, threshold: float = 0.0035) -> str:
        """Categorize recent volatility as 'high' or 'low'."""
        vol = self.volatility(index, window=window)
        return "high" if vol >= threshold else "low"

    def trend_persistence(self, index: int, window: int = 24) -> float:
        """Fraction of consecutive price moves that continue in the same direction."""
        start = max(0, index - window)
        window_closes = self.closes[start:index + 1]
        if len(window_closes) < 3:
            return 0.5
        diffs = [b - a for a, b in zip(window_closes, window_closes[1:])]
        same_dir = sum(a * b > 0 for a, b in zip(diffs, diffs[1:]))
        return same_dir / max(1, len(diffs) - 1)

    def trend_regime(self, index: int, window: int = 24, threshold: float = 0.45) -> str:
        """Categorize market state as 'trending' (directional momentum) or 'chop' (ranging/reverting)."""
        persist = self.trend_persistence(index, window=window)
        return "trending" if persist >= threshold else "chop"


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

    # CSV headers are singular (open/high/low/close/volume), while the
    # in-memory dataset uses plural attribute names (opens/highs/...).
    column_names = {
        "timestamps": timestamp_name,
        "opens": names["open"],
        "highs": names["high"],
        "lows": names["low"],
        "closes": names["close"],
        "volumes": names["volume"],
    }
    columns = {key: [] for key in column_names}
    for row in rows:
        columns["timestamps"].append(int(float(row[column_names["timestamps"]])))
        for key in ("opens", "highs", "lows", "closes", "volumes"):
            columns[key].append(float(row[column_names[key]]))

    timestamps = columns["timestamps"]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise ValueError("BNB data must be strictly chronological")
    return BNBPredictionDataset(**{key: tuple(value) for key, value in columns.items()})


def download_bnb_5m_csv(path: str | Path, limit: int = 1000) -> Path:
    if limit < 2:
        raise ValueError("limit must be at least two")
    from urllib.parse import urlencode
    import json

    all_rows = []
    remaining = limit
    end_time = None

    while remaining > 0:
        batch_limit = min(1000, remaining)
        params: dict[str, str | int] = {
            "symbol": "BNBUSDT",
            "interval": "5m",
            "limit": batch_limit,
        }
        if end_time is not None:
            params["endTime"] = end_time
        query = urlencode(params)
        with urlopen(f"{BINANCE_KLINES_URL}?{query}", timeout=30) as response:
            batch = json.load(response)
        if not batch:
            break
        all_rows = batch + all_rows
        remaining -= len(batch)
        first_open_time = int(batch[0][0])
        end_time = first_open_time - 1
        if len(batch) < batch_limit:
            break

    # Deduplicate and sort chronologically
    seen = set()
    deduped = []
    for row in sorted(all_rows, key=lambda r: int(r[0])):
        ts = int(row[0])
        if ts not in seen:
            seen.add(ts)
            deduped.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in deduped:
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
