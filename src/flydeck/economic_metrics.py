from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class EconomicSurvivalMetrics:
    net_return: float
    gross_return: float
    volatility: float
    max_drawdown: float
    return_over_drawdown: float
    sharpe_net: float
    cvar_95: float
    cvar_99: float
    profit_factor: float
    ruin_probability: float
    recovery_periods: int
    turnover: float
    total_cost: float
    average_exposure: float
    worst_period_return: float


def calculate_economic_survival_metrics(
    returns: list[float] | tuple[float, ...],
    positions: list[float] | tuple[float, ...] = (),
    *,
    costs: list[float] | tuple[float, ...] = (),
    ruin_barrier: float = 0.70,
    annualization_periods: float = 288.0,
) -> EconomicSurvivalMetrics:
    net = [float(value) for value in returns]
    gross = [value + (costs[i] if i < len(costs) else 0.0) for i, value in enumerate(net)]
    equity = 1.0
    peak = equity
    max_drawdown = 0.0
    recovery_periods = 0
    underwater = 0
    ruin_hits = 0
    for value in net:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = (peak - equity) / max(1e-12, peak)
        max_drawdown = max(max_drawdown, drawdown)
        if equity <= ruin_barrier:
            ruin_hits += 1
        if equity < peak:
            underwater += 1
            recovery_periods = max(recovery_periods, underwater)
        else:
            underwater = 0
    mean = sum(net) / len(net) if net else 0.0
    variance = sum((value - mean) ** 2 for value in net) / len(net) if net else 0.0
    std = math.sqrt(max(0.0, variance))
    gains = sum(value for value in net if value > 0.0)
    losses = -sum(value for value in net if value < 0.0)
    sorted_losses = sorted((value for value in net if value < 0.0))
    cvar_95 = _tail_mean(sorted_losses, max(1, math.ceil(len(sorted_losses) * 0.05)))
    cvar_99 = _tail_mean(sorted_losses, max(1, math.ceil(len(sorted_losses) * 0.01)))
    total_cost = sum(costs) if costs else 0.0
    turnover = sum(abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))) if len(positions) > 1 else 0.0
    net_return = _compound(net)
    gross_return = _compound(gross)
    return EconomicSurvivalMetrics(
        net_return=net_return,
        gross_return=gross_return,
        volatility=std,
        max_drawdown=max_drawdown,
        return_over_drawdown=net_return / max_drawdown if max_drawdown > 1e-12 else 0.0,
        sharpe_net=mean / std * math.sqrt(annualization_periods) if std > 1e-12 else 0.0,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        profit_factor=gains / losses if losses > 1e-12 else (math.inf if gains > 0 else 0.0),
        ruin_probability=ruin_hits / len(net) if net else 0.0,
        recovery_periods=recovery_periods,
        turnover=turnover,
        total_cost=total_cost,
        average_exposure=sum(abs(value) for value in positions) / len(positions) if positions else 0.0,
        worst_period_return=min(net) if net else 0.0,
    )


def _compound(values: list[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _tail_mean(values: list[float], count: int) -> float:
    return sum(values[:count]) / count if values else 0.0
