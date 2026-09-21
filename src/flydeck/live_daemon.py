"""Continuous live prediction daemon for FlyDeck Agent.

Runs 24/7 synchronized with the BNB 5-minute candle boundary:
- Fetches newly finalized candles from Binance Vision via canonical MarketDataService.
- Updates causal Mushroom Body synaptic reinforcement from prior round.
- Emits real-time UP, DOWN, or WAIT prediction with explicit diagnostic reasoning.
- Saves atomic checkpoints and writes auditable JSONL logs.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .checkpoint import AgentCheckpointManager
from .data import BinanceMarketDataProvider, MarketDataService
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


class FlyDeckLiveDaemon:
    """Orchestrates 24/7 live inference, learning, and persistence."""

    def __init__(
        self,
        agent: FlyVisualPredictionAgent,
        symbol: str = "BNBUSDT",
        interval: str = "5m",
        interval_seconds: int = 300,
        checkpoint_file: str | Path = "data/checkpoints/live_fly_brain.json",
        diagnostics_file: str | Path = "data/logs/live_diagnostics.jsonl",
        context_window: int = 32,
        service: MarketDataService | None = None,
        on_action: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.agent = agent
        self.symbol = symbol.upper()
        self.interval = interval
        self.interval_seconds = interval_seconds
        self.checkpoint_file = Path(checkpoint_file)
        self.diagnostics_file = Path(diagnostics_file)
        self.context_window = context_window
        self.service = service or MarketDataService(BinanceMarketDataProvider())
        self.on_action = on_action

        self.running = False
        self.round_count = 0
        self.last_candle_timestamp: int | None = None
        self._prev_close: float | None = None

        # Try to restore checkpoint if available
        if self.checkpoint_file.exists():
            try:
                meta = AgentCheckpointManager.load(self.agent, self.checkpoint_file)
                self.round_count = meta.get("round_count", 0)
                print(f"[FlyDeck Daemon] Restored checkpoint from {self.checkpoint_file} (round {self.round_count})")
            except Exception as e:
                print(f"[FlyDeck Daemon] Warning: Failed to load existing checkpoint: {e}")

    def stop(self) -> None:
        """Gracefully stop the daemon and save state."""
        self.running = False

    def time_to_next_boundary(self, current_time: float | None = None, buffer_seconds: float = 2.5) -> float:
        """Calculate wait seconds until the next 5-minute candle close."""
        now = current_time if current_time is not None else time.time()
        next_boundary = (int(now // self.interval_seconds) + 1) * self.interval_seconds
        return max(0.5, (next_boundary - now) + buffer_seconds)

    def execute_step(self, current_time: float | None = None) -> dict[str, Any] | None:
        """Execute one evaluation round upon candle close."""
        # 1. Fetch recent candles
        dataset = self.service.fetch(self.symbol, self.interval, limit=self.context_window + 5)
        if not dataset or dataset.size < self.context_window:
            print(f"[FlyDeck Daemon] Insufficient candles fetched: {dataset.size if dataset else 0}")
            return None

        latest_candle = dataset.candles[-1]
        # Ignore if we already processed this exact candle
        if self.last_candle_timestamp is not None and latest_candle.timestamp <= self.last_candle_timestamp:
            return None

        self.last_candle_timestamp = latest_candle.timestamp
        self.round_count += 1

        # 2. Extract window
        prices = dataset.closes[-self.context_window :]
        volumes = dataset.volumes[-self.context_window :]

        # 3. Perceive & Decide (visual agent handles MB causal reinforcement internally using current_price)
        stimulus, decision = self.agent.perceive(prices, volumes=volumes)
        state = self.agent.internal_state

        # Compute realized return from previous round if available
        observed_return = None
        if self._prev_close is not None and self._prev_close > 0:
            observed_return = (latest_candle.close / self._prev_close - 1.0) * 100.0
        self._prev_close = latest_candle.close

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 4. Formulate record
        record: dict[str, Any] = {
            "round": self.round_count,
            "timestamp": latest_candle.timestamp,
            "datetime_utc": now_utc,
            "close": latest_candle.close,
            "action": decision.action.name,
            "reason": decision.reason,
            "confidence": round(decision.confidence, 4),
            "up_score": round(decision.up_score, 4),
            "down_score": round(decision.down_score, 4),
            "conflict": round(decision.conflict, 4),
            "arousal": round(state.arousal, 4),
            "cx_heading": round(state.cx_heading, 4),
            "mb_valence": round(state.mb_valence, 4),
            "expectation": round(state.expectation, 4),
            "observed_return_prev": round(observed_return, 4) if observed_return is not None else None,
        }

        # 5. Save atomic checkpoint
        metadata = {
            "round_count": self.round_count,
            "last_timestamp": latest_candle.timestamp,
            "last_close": latest_candle.close,
            "last_action": decision.action.name,
            "saved_at_utc": now_utc,
        }
        AgentCheckpointManager.save(self.agent, self.checkpoint_file, metadata=metadata)

        # 6. Append to diagnostics JSONL
        self.diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
        with self.diagnostics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        # 7. Print formatted log line
        action_colored = record["action"]
        prev_ret_str = f"{observed_return:+.2f}%" if observed_return is not None else "N/A"
        print(
            f"[{now_utc}] Round #{self.round_count:<4} | Close: {latest_candle.close:<8.2f} | "
            f"Action: {action_colored:<11} | Conf: {record['confidence']:<6.2f} | "
            f"Conflict: {record['conflict']:<5.2f} | MB Val: {record['mb_valence']:<6.2f} | "
            f"Prev Ret: {prev_ret_str} -> Checkpoint Saved"
        )

        # 8. Trigger external execution hook if registered
        if self.on_action is not None:
            try:
                self.on_action(record["action"], record)
            except Exception as e:
                print(f"[FlyDeck Daemon] Erro no callback de ação (on_action): {e}")

        return record

    def run(self, max_rounds: int | None = None) -> None:
        """Run the continuous 24/7 daemon loop."""
        self.running = True
        print(f"[FlyDeck Daemon] Starting continuous run for {self.symbol} ({self.interval})...")
        print(f"[FlyDeck Daemon] Checkpoints: {self.checkpoint_file} | Logs: {self.diagnostics_file}")

        rounds_completed = 0
        try:
            while self.running:
                sleep_secs = self.time_to_next_boundary()
                time.sleep(sleep_secs)

                if not self.running:
                    break

                try:
                    res = self.execute_step()
                    if res is not None:
                        rounds_completed += 1
                        if max_rounds is not None and rounds_completed >= max_rounds:
                            break
                except Exception as e:
                    print(f"[FlyDeck Daemon] Error in step execution: {e}")
                    time.sleep(5.0)  # Brief backoff on network error

        except KeyboardInterrupt:
            print("\n[FlyDeck Daemon] KeyboardInterrupt caught. Shutting down gracefully...")
        finally:
            self.running = False
            # Final checkpoint save
            meta = {
                "round_count": self.round_count,
                "stopped_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            AgentCheckpointManager.save(self.agent, self.checkpoint_file, metadata=meta)
            print("[FlyDeck Daemon] Final checkpoint persisted. Daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyDeck Live Prediction Daemon (24/7)")
    parser.add_argument("--circuit", default="data/malecns/motion_visual.json", type=Path, help="Path to visual circuit JSON")
    parser.add_argument("--symbol", default="BNBUSDT", help="Pair to trade")
    parser.add_argument("--interval", default="5m", help="Candle interval")
    parser.add_argument("--checkpoint", default="data/checkpoints/live_fly_brain.json", help="Checkpoint file path")
    parser.add_argument("--logs", default="data/logs/live_diagnostics.jsonl", help="Diagnostics log path")
    parser.add_argument("--confidence", default=0.15, type=float, help="Base confidence threshold")
    parser.add_argument("--max-rounds", type=int, default=None, help="Stop after N rounds (default: run forever)")
    args = parser.parse_args()

    circuit = VisualCircuit.load(args.circuit)
    agent = FlyVisualPredictionAgent(
        circuit,
        retina_width=32,
        retina_height=16,
        confidence_threshold=args.confidence,
    )

    daemon = FlyDeckLiveDaemon(
        agent=agent,
        symbol=args.symbol,
        interval=args.interval,
        checkpoint_file=args.checkpoint,
        diagnostics_file=args.logs,
    )

    daemon.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
