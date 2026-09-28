from __future__ import annotations

import argparse
from pathlib import Path

from .pancakeswap_rpc import BNB_PREDICTION_CONTRACT, PancakeRpcClient, write_pancake_rounds_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PancakeSwap BNB Prediction round data through read-only JSON-RPC"
    )
    parser.add_argument("--rpc-url", required=True, help="BNB Chain JSON-RPC HTTPS endpoint")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV")
    parser.add_argument("--start-epoch", required=True, type=int)
    parser.add_argument(
        "--end-epoch",
        type=int,
        help="Inclusive end epoch; defaults to currentEpoch - 1",
    )
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--contract", default=BNB_PREDICTION_CONTRACT)
    args = parser.parse_args()

    client = PancakeRpcClient(args.rpc_url, contract_address=args.contract)
    end_epoch = args.end_epoch
    if end_epoch is None:
        end_epoch = max(args.start_epoch, client.current_epoch() - 1)

    print(f"Fetching PancakeSwap BNB rounds {args.start_epoch}..{end_epoch}...")
    rounds = client.fetch_rounds(args.start_epoch, end_epoch, batch_size=args.batch_size)
    destination = write_pancake_rounds_csv(rounds, args.output)
    settled = sum(row.oracle_called for row in rounds)
    print(f"Saved {len(rounds)} rounds ({settled} oracle-settled) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
