from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from .bnb_prediction import BNBPredictionRound, Prediction

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

    def observation(self, index: int, context: int = 24):
        start = max(0, index - context + 1)
        from .bnb_prediction_agent import BNBObservation
        return BNBObservation(
            timestamp=self.timestamps[index],
            prices=self.closes[start : index + 1],
            volumes=self.volumes[start : index + 1],
        )

    def round(self, index: int) -> BNBPredictionRound:
        return BNBPredictionRound(
            timestamp=self.timestamps[index],
            reference_price=self.closes[index],
            close_price=self.closes[index + 1],
        )


def load_bnb_5m_csv(path: str | Path) -> BNBPredictionDataset:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("BNB CSV is empty")
    names = {name.strip().lower(): name for name in rows[0]}
    required = {"open", "high", "low", "close", "volume"}
    missing = required - names.keys()
    timestamp_name = names.get("timestamp") or names.get("open_time") or names.get("open time")
    if timestamp_name is None:
        missing.add("timestamp")
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")

    columns = {key: [] for key in ("timestamps", "opens", "highs", "lows", "closes", "volumes")}
    for row in rows:
        columns["timestamps"].append(int(float(row[timestamp_name])))
        for key in ("opens", "highs", "lows", "closes", "volumes"):
            columns[key].append(float(row[names[key]]))
    if any(columns["timestamps"][i] >= columns["timestamps"][i + 1] for i in range(len(rows) - 1)):
        raise ValueError("BNB data must be strictly chronological")
    return BNBPredictionDataset(**{key: tuple(value) for key, value in columns.items()})


def download_bnb_5m_csv(path: str | Path, limit: int = 1000) -> Path:
    """Download the latest BNBUSDT 5m candles from Binance public market data."""
    if limit < 2 or limit > 1000:
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
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
    return path


def outcome_for_prices(reference: float, future: float) -> Prediction:
    if future > reference:
        return Prediction.UP
    if future < reference:
        return Prediction.DOWN
    return Prediction.WAIT
