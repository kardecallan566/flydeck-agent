from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from .pancakeswap_targets import PancakeRound


BNB_PREDICTION_CONTRACT = "0x18b2a687610328590bc8f2e5fedde3b582a49cda"
_CURRENT_EPOCH_SELECTOR = "0x76671808"
_ROUNDS_SELECTOR = "0x8c65c81f"


class PancakeRpcClient:
    """Minimal read-only JSON-RPC client for the official BNB Prediction V2 contract."""

    def __init__(
        self,
        rpc_url: str,
        *,
        contract_address: str = BNB_PREDICTION_CONTRACT,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not rpc_url:
            raise ValueError("rpc_url is required")
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.timeout_seconds = timeout_seconds

    def current_epoch(self) -> int:
        result = self._single_eth_call(_CURRENT_EPOCH_SELECTOR)
        return int(result, 16)

    def fetch_rounds(
        self,
        start_epoch: int,
        end_epoch: int,
        *,
        batch_size: int = 100,
    ) -> tuple[PancakeRound, ...]:
        if start_epoch < 0 or end_epoch < start_epoch:
            raise ValueError("invalid epoch range")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        rows: list[PancakeRound] = []
        for batch_start in range(start_epoch, end_epoch + 1, batch_size):
            epochs = list(range(batch_start, min(end_epoch + 1, batch_start + batch_size)))
            payload = []
            for request_id, epoch in enumerate(epochs, start=1):
                payload.append(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "eth_call",
                        "params": [
                            {
                                "to": self.contract_address,
                                "data": _round_call_data(epoch),
                            },
                            "latest",
                        ],
                    }
                )
            response = self._post(payload)
            if not isinstance(response, list):
                raise RuntimeError("RPC batch response is not a list")
            by_id = {int(item["id"]): item for item in response}
            for request_id, epoch in enumerate(epochs, start=1):
                item = by_id.get(request_id)
                if item is None:
                    raise RuntimeError(f"RPC response missing epoch {epoch}")
                if "error" in item:
                    raise RuntimeError(f"RPC error for epoch {epoch}: {item['error']}")
                rows.append(decode_bnb_prediction_round(str(item["result"])))
        return tuple(rows)

    def _single_eth_call(self, data: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": self.contract_address, "data": data}, "latest"],
        }
        response = self._post(payload)
        if not isinstance(response, dict):
            raise RuntimeError("RPC response is not an object")
        if "error" in response:
            raise RuntimeError(f"RPC error: {response['error']}")
        return str(response["result"])

    def _post(self, payload: object) -> object:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.rpc_url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "flydeck-agent/0.1"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def decode_bnb_prediction_round(encoded: str) -> PancakeRound:
    """Decode rounds(epoch) for the official BNBUSD Prediction V2 contract."""
    raw = encoded[2:] if encoded.startswith("0x") else encoded
    if len(raw) < 14 * 64:
        raise ValueError("rounds(epoch) RPC result is shorter than the expected 14 ABI words")
    words = [int(raw[offset : offset + 64], 16) for offset in range(0, 14 * 64, 64)]
    lock_price = _signed_256(words[4])
    close_price = _signed_256(words[5])
    return PancakeRound(
        epoch=words[0],
        start_timestamp_ms=words[1] * 1000,
        lock_timestamp_ms=words[2] * 1000,
        close_timestamp_ms=words[3] * 1000,
        lock_price=float(lock_price),
        close_price=float(close_price),
        oracle_called=bool(words[13]),
        total_amount=float(words[8]),
        bull_amount=float(words[9]),
        bear_amount=float(words[10]),
        reward_base_cal_amount=float(words[11]),
        reward_amount=float(words[12]),
    )


def write_pancake_rounds_csv(rounds: Iterable[PancakeRound], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "epoch",
                "startTimestamp",
                "lockTimestamp",
                "closeTimestamp",
                "lockPrice",
                "closePrice",
                "totalAmount",
                "bullAmount",
                "bearAmount",
                "rewardBaseCalAmount",
                "rewardAmount",
                "oracleCalled",
            )
        )
        for row in rounds:
            writer.writerow(
                (
                    row.epoch,
                    row.start_timestamp_ms // 1000,
                    row.lock_timestamp_ms // 1000,
                    row.close_timestamp_ms // 1000,
                    _format_number(row.lock_price),
                    _format_number(row.close_price),
                    _format_optional(row.total_amount),
                    _format_optional(row.bull_amount),
                    _format_optional(row.bear_amount),
                    _format_optional(row.reward_base_cal_amount),
                    _format_optional(row.reward_amount),
                    str(row.oracle_called).lower(),
                )
            )
    return destination


def _round_call_data(epoch: int) -> str:
    return _ROUNDS_SELECTOR + f"{epoch:064x}"


def _signed_256(value: int) -> int:
    return value - (1 << 256) if value >= (1 << 255) else value


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _format_optional(value: float | None) -> str:
    return "" if value is None else _format_number(value)
