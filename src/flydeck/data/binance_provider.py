from __future__ import annotations

import json
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .market_data import MarketCandle, MarketDataProvider


class BinanceMarketDataProvider(MarketDataProvider):
    """Public Binance Vision REST provider with bounded retries and pagination."""

    name = "binance"
    base_url = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.75) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _request(self, params: dict[str, str | int]) -> list[list[object]]:
        query = urlencode(params)
        request = Request(
            f"{self.base_url}?{query}",
            headers={"User-Agent": "flydeck-agent/0.1"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, list):
                    raise ValueError("Binance returned a non-list payload")
                return payload
            except Exception as exc:  # network/API failures are retried at the provider boundary
                last_error = exc
                if attempt < self.retries:
                    sleep(self.backoff * (2**attempt))
        assert last_error is not None
        raise RuntimeError(f"Binance market-data request failed: {last_error}") from last_error

    def fetch_range(
        self, symbol: str, interval: str, start_ms: int, stop_ms: int,
    ) -> tuple[MarketCandle, ...]:
        """Fetch [start_ms, stop_ms) in ascending order, including older data.

        Unlike fetch(limit), this is anchored to exact UTC candle timestamps.
        A caller must still verify that every expected interval is present.
        """
        if interval != "5m":
            raise ValueError("bounded gap repair currently requires 5m candles")
        step = 300_000
        if (start_ms < 0 or stop_ms <= start_ms
            or start_ms % step or stop_ms % step):
            raise ValueError("range bounds must be aligned, positive 5-minute UTC timestamps")
        if (stop_ms - start_ms) // step > 100_000:
            raise ValueError("requested range exceeds 100,000 5-minute candles")
        output: dict[int, MarketCandle] = {}
        cursor = start_ms
        while cursor < stop_ms:
            remaining = (stop_ms - cursor) // step
            batch = self._request({
                "symbol": symbol.upper(), "interval": interval,
                "startTime": cursor, "endTime": stop_ms - 1,
                "limit": min(1000, remaining),
            })
            if not batch:
                raise ValueError(
                    f"Binance returned no candles in the requested range starting at {cursor}"
                )
            for row in batch:
                ts = int(row[0])
                if start_ms <= ts < stop_ms:
                    output[ts] = MarketCandle(
                        timestamp=ts, open=float(row[1]),
                        high=float(row[2]), low=float(row[3]),
                        close=float(row[4]), volume=float(row[5]),
                    )
            newest = max(int(row[0]) for row in batch)
            if newest < cursor:
                raise ValueError("Binance returned non-progressing timestamps")
            cursor = newest + step
        return tuple(output[t] for t in sorted(output))

    def fetch(self, symbol: str, interval: str, limit: int = 1000) -> tuple[MarketCandle, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if limit > 100_000:
            raise ValueError("limit is capped at 100000 candles per request")

        rows: list[list[object]] = []
        remaining = limit
        end_time: int | None = None

        while remaining > 0:
            batch_limit = min(1000, remaining)
            params: dict[str, str | int] = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": batch_limit,
            }
            if end_time is not None:
                params["endTime"] = end_time
            batch = self._request(params)
            if not batch:
                break
            rows = batch + rows
            remaining -= len(batch)
            first_open_time = int(batch[0][0])
            end_time = first_open_time - 1
            if len(batch) < batch_limit:
                break

        candles = [
            MarketCandle(
                timestamp=int(float(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        ]
        unique = {candle.timestamp: candle for candle in candles}
        return tuple(unique[timestamp] for timestamp in sorted(unique))
