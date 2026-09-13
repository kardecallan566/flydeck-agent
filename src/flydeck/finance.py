from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .environment import StepResult


@dataclass(frozen=True, slots=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class FinancialEpisodeResult:
    total_reward: float
    final_portfolio: float
    return_pct: float
    max_drawdown_pct: float
    trades: int
    steps: int


class SyntheticCryptoMarket:
    """Generate unpredictable but reproducible crypto-like OHLCV episodes."""

    REGIMES = ("trend_up", "trend_down", "sideways", "volatile", "reversal")

    def __init__(
        self,
        length: int = 256,
        initial_price: float = 100.0,
        seed: int = 42,
    ) -> None:
        if length < 32:
            raise ValueError("market length must be >= 32")
        if initial_price <= 0:
            raise ValueError("initial_price must be > 0")
        self.length = length
        self.initial_price = initial_price
        self.seed = seed

    def generate(self) -> tuple[Candle, ...]:
        rng = random.Random(self.seed)
        price = self.initial_price
        candles: list[Candle] = []
        regime = rng.choice(self.REGIMES)
        regime_steps = 0
        previous_return = 0.0

        for index in range(self.length):
            if regime_steps <= 0:
                regime = rng.choice(self.REGIMES)
                regime_steps = rng.randint(12, 48)

            if regime == "trend_up":
                drift, volatility, persistence = 0.0025, 0.008, 0.25
            elif regime == "trend_down":
                drift, volatility, persistence = -0.0025, 0.008, 0.25
            elif regime == "sideways":
                drift, volatility, persistence = 0.0, 0.006, -0.10
            elif regime == "volatile":
                drift, volatility, persistence = 0.0, 0.025, 0.10
            else:  # reversal
                drift = -0.003 if previous_return > 0 else 0.003
                volatility, persistence = 0.012, -0.35

            shock = 0.0
            if rng.random() < 0.025:
                shock = rng.gauss(0.0, volatility * 4.0)

            log_return = (
                drift
                + persistence * previous_return
                + rng.gauss(0.0, volatility)
                + shock
            )
            log_return = max(-0.25, min(0.25, log_return))

            open_price = price
            close = price * math.exp(log_return)
            intraday = abs(log_return) + volatility * rng.uniform(0.5, 1.5)
            high = max(open_price, close) * math.exp(rng.uniform(0.0, intraday))
            low = min(open_price, close) * math.exp(-rng.uniform(0.0, intraday))
            volume = 1000.0 * (1.0 + abs(log_return) * 18.0) * rng.uniform(0.7, 1.3)

            candles.append(Candle(open_price, high, low, close, volume))
            price = close
            previous_return = log_return
            regime_steps -= 1

        return tuple(candles)


class CryptoTradingEnvironment:
    """Long-only sequential trading environment with BUY/HOLD/SELL actions.

    Actions: 0=HOLD, 1=BUY, 2=SELL. Each trade changes the asset allocation
    by 25 percentage points and pays a configurable transaction fee.
    """

    ACTIONS = ("hold", "buy", "sell")
    observation_size = 12
    action_size = 3

    def __init__(
        self,
        candles: tuple[Candle, ...] | list[Candle],
        initial_capital: float = 10_000.0,
        trade_fraction: float = 0.25,
        fee_rate: float = 0.001,
        window: int = 24,
        max_steps: int | None = None,
    ) -> None:
        if len(candles) < window + 2:
            raise ValueError("candles must contain at least window + 2 entries")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0 < trade_fraction <= 1:
            raise ValueError("trade_fraction must be in (0, 1]")
        if fee_rate < 0:
            raise ValueError("fee_rate must be >= 0")
        self.candles = tuple(candles)
        self.initial_capital = initial_capital
        self.trade_fraction = trade_fraction
        self.fee_rate = fee_rate
        self.window = window
        self.max_steps = max_steps or len(self.candles) - window - 1
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.index = window
        self.steps = 0
        self.cash = initial_capital
        self.asset_units = 0.0
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.trades = 0
        self._portfolio_value = initial_capital

    def reset(self) -> tuple[float, ...]:
        self.index = self.window
        self.steps = 0
        self.cash = self.initial_capital
        self.asset_units = 0.0
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.trades = 0
        self._portfolio_value = self.initial_capital
        return self._observation()

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.action_size:
            raise ValueError("action is outside the environment action range")

        price = self.candles[self.index].close
        traded = False
        if action == 1:
            amount = self.cash * self.trade_fraction
            if amount > 0:
                cost = amount * (1.0 + self.fee_rate)
                if cost <= self.cash:
                    self.cash -= cost
                    self.asset_units += amount / price
                    traded = True
        elif action == 2:
            amount_units = self.asset_units * self.trade_fraction
            if amount_units > 0:
                proceeds = amount_units * price
                self.asset_units -= amount_units
                self.cash += proceeds * (1.0 - self.fee_rate)
                traded = True

        if traded:
            self.trades += 1

        previous_value = self._portfolio_value
        self.index += 1
        self.steps += 1
        next_price = self.candles[self.index].close
        self._portfolio_value = self.cash + self.asset_units * next_price
        reward = self._portfolio_value / previous_value - 1.0

        self.peak_value = max(self.peak_value, self._portfolio_value)
        drawdown = self._portfolio_value / self.peak_value - 1.0
        self.max_drawdown = min(self.max_drawdown, drawdown)

        done = self.steps >= self.max_steps or self.index >= len(self.candles) - 1
        return StepResult(self._observation(), reward, done)

    def episode_result(self, total_reward: float) -> FinancialEpisodeResult:
        return FinancialEpisodeResult(
            total_reward=total_reward,
            final_portfolio=self._portfolio_value,
            return_pct=(self._portfolio_value / self.initial_capital - 1.0) * 100.0,
            max_drawdown_pct=self.max_drawdown * 100.0,
            trades=self.trades,
            steps=self.steps,
        )

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def position_ratio(self) -> float:
        value = self._portfolio_value
        return 0.0 if value <= 0 else (self.asset_units * self.candles[self.index].close) / value

    def _observation(self) -> tuple[float, ...]:
        current = self.candles[self.index]
        closes = [candle.close for candle in self.candles]
        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]

        def period_return(period: int) -> float:
            start = max(0, self.index - period)
            return closes[self.index] / closes[start] - 1.0

        def volatility(period: int) -> float:
            values = returns[max(0, self.index - period) : self.index]
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return math.sqrt(variance)

        recent = closes[max(0, self.index - self.window) : self.index + 1]
        low = min(recent)
        high = max(recent)
        range_position = 0.5 if high == low else (current.close - low) / (high - low)
        avg_volume = sum(c.volume for c in self.candles[max(0, self.index - 20) : self.index]) / max(
            1, min(20, self.index)
        )

        return (
            max(-1.0, min(1.0, period_return(1) * 20.0)),
            max(-1.0, min(1.0, period_return(3) * 10.0)),
            max(-1.0, min(1.0, period_return(6) * 7.0)),
            max(-1.0, min(1.0, period_return(12) * 5.0)),
            max(-1.0, min(1.0, period_return(24) * 3.0)),
            min(1.0, volatility(6) * 50.0),
            min(1.0, volatility(24) * 25.0),
            max(0.0, min(2.0, current.volume / max(avg_volume, 1.0))),
            range_position,
            self.position_ratio,
            self.cash / max(self._portfolio_value, 1e-12),
            max(-1.0, min(1.0, (self._portfolio_value / self.initial_capital - 1.0))),
        )
