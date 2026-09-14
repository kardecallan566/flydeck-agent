from __future__ import annotations

from dataclasses import dataclass

from .finance import Candle, CryptoTradingEnvironment, SyntheticCryptoMarket
from .malecns import MaleCNSCircuit, MaleCNSReservoir


@dataclass(frozen=True, slots=True)
class MaleCNSFinanceResult:
    episodes: int
    average_return_pct: float
    best_return_pct: float
    last_return_pct: float
    average_drawdown_pct: float
    total_trades: int
    hold_actions: int
    buy_actions: int
    sell_actions: int

    @property
    def total_actions(self) -> int:
        return self.hold_actions + self.buy_actions + self.sell_actions

    @property
    def trade_frequency(self) -> float:
        return self.total_trades / max(1, self.total_actions)


def train_malecns_synthetic(
    circuit: MaleCNSCircuit,
    *,
    episodes: int = 20,
    market_length: int = 256,
    max_steps: int = 200,
    seed: int = 42,
    epsilon: float = 0.20,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.05,
    discount: float = 0.97,
    learning_rate: float = 0.005,
) -> MaleCNSFinanceResult:
    """Trade synthetic crypto using a frozen MaleCNS reservoir + trainable readout."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    reservoir = MaleCNSReservoir(circuit, feature_count=12, action_count=3, seed=seed)
    returns: list[float] = []
    drawdowns: list[float] = []
    counts = [0, 0, 0]
    total_trades = 0
    current_epsilon = epsilon

    for episode in range(episodes):
        candles = SyntheticCryptoMarket(length=market_length, seed=seed + episode).generate()
        environment = CryptoTradingEnvironment(
            candles,
            max_steps=max_steps,
            trade_penalty=0.0025,
            invalid_action_penalty=0.001,
            drawdown_penalty=0.02,
        )
        observation = environment.reset()
        reservoir.reset()
        scores = reservoir.step(observation)
        total_reward = 0.0

        for _ in range(max_steps):
            action = _choose_action(scores, reservoir, current_epsilon)
            counts[action] += 1

            result = environment.step(action)
            total_reward += result.reward
            if result.done:
                reservoir.update_readout(action, result.reward, learning_rate)
                break

            next_scores = reservoir.step(result.observation)
            current = scores[action]
            target = result.reward + discount * max(next_scores)
            td_error = max(-1.0, min(1.0, target - current))
            reservoir.update_readout(action, td_error, learning_rate)
            scores = next_scores

        metrics = environment.episode_result(total_reward)
        returns.append(metrics.return_pct)
        drawdowns.append(metrics.max_drawdown_pct)
        total_trades += metrics.trades
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return MaleCNSFinanceResult(
        episodes=episodes,
        average_return_pct=sum(returns) / len(returns),
        best_return_pct=max(returns),
        last_return_pct=returns[-1],
        average_drawdown_pct=sum(drawdowns) / len(drawdowns),
        total_trades=total_trades,
        hold_actions=counts[0],
        buy_actions=counts[1],
        sell_actions=counts[2],
    )


def _choose_action(scores: tuple[float, ...], reservoir: MaleCNSReservoir, epsilon: float) -> int:
    # Reservoir owns the readout but not the exploration policy; use its scores
    # directly so the experiment remains comparable to the existing finance agent.
    if reservoir.action_count != len(scores):
        raise ValueError("score/action count mismatch")
    # Deterministic exploration is handled by the caller's global seed through
    # the reservoir instance; this helper intentionally uses greedy selection.
    return max(range(len(scores)), key=scores.__getitem__)
