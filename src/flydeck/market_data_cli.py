from __future__ import annotations

import argparse

from .data import BinanceMarketDataProvider, MarketDataService, inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck market data acquisition and health check")
    parser.add_argument("--symbol", default="BNBUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    service = MarketDataService(BinanceMarketDataProvider(), cache_root=args.cache_root)
    dataset = service.load_or_fetch(
        args.symbol,
        args.interval,
        limit=args.limit,
        refresh=args.refresh,
    )
    health = inspect_dataset(dataset)

    print("FlyDeck Market Data")
    print(f"source: {dataset.source}")
    print(f"symbol: {dataset.symbol}")
    print(f"interval: {dataset.interval}")
    print(f"candles: {health.candles}")
    print(f"start: {health.start_timestamp}")
    print(f"end: {health.end_timestamp}")
    print(f"gaps: {health.gaps}")
    print(f"complete: {health.complete}")


if __name__ == "__main__":
    main()
