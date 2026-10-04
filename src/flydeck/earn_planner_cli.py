"""Offline PancakeSwap economics gate: honest risk and break-even illustrations.

NO wallet, no trades, no token recommendations, no live/APR scraping.
Only user-supplied hypothetical values; ALWAYS report zero as a valid choice.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LpScenario:
    capital_usd: float
    token_a_change_pct: float
    token_b_change_pct: float = 0.0
    swap_fee_apr_pct: float = 0.0
    farm_apr_pct: float = 0.0
    days: int = 30
    total_gas_usd: float = 0.0
    slippage_usd: float = 0.0

    def __post_init__(self):
        for name, number in asdict(self).items():
            if isinstance(number, bool) or not math.isfinite(float(number)):
                raise ValueError(f"{name} must be a finite number")
        if self.capital_usd <= 0 or self.days <= 0:
            raise ValueError("capital and time horizon must be positive")
        if self.token_a_change_pct <= -100 or self.token_b_change_pct <= -100:
            raise ValueError("asset change cannot be <= -100% in this illustration")
        if not 0 <= self.swap_fee_apr_pct <= 1000 or not 0 <= self.farm_apr_pct <= 1000:
            raise ValueError("APR assumptions must lie between 0 and 1000%")
        if self.total_gas_usd < 0 or self.slippage_usd < 0:
            raise ValueError("costs must not be negative")


def simulate_v2_equal_weight(s: LpScenario) -> dict:
    """Illustration ONLY for a fully invested, frictionless constant-product 50:50 V2.

    Assumes fee/farm APR accrues linearly on initial USD capital for a specified
    horizon. Does not model concentrated liquidity V3, gas variability, reward
    token price changes, pool hacks, peg risk or realistic compounding.
    """
    a = 1 + s.token_a_change_pct / 100
    b = 1 + s.token_b_change_pct / 100
    ratio = a / b
    hold_value = s.capital_usd * (a + b) / 2
    lp_value = s.capital_usd * math.sqrt(a * b)
    il = 2 * math.sqrt(ratio) / (1 + ratio) - 1
    fees = s.capital_usd * (s.swap_fee_apr_pct / 100) * s.days / 365
    farming = s.capital_usd * (s.farm_apr_pct / 100) * s.days / 365
    costs = s.total_gas_usd + s.slippage_usd
    net_lp = lp_value + fees + farming - costs
    difference = net_lp - hold_value
    return {
        "research_only": True, "requires_capital_and_risks_loss": True,
        "model": "constant_product_v2_50_50_only",
        "inputs": asdict(s),
        "hypothetical_hold_value_usd": round(hold_value, 8),
        "hypothetical_lp_excluding_yield_usd": round(lp_value, 8),
        "impermanent_loss_relative_to_hold_pct": round(100 * il, 8),
        "illustrative_fee_revenue_usd": round(fees, 8),
        "illustrative_farm_rewards_usd": round(farming, 8),
        "assumed_gas_and_slippage_usd": round(costs, 8),
        "illustrative_net_lp_value_usd": round(net_lp, 8),
        "lp_minus_hold_usd": round(difference, 8),
        "net_lp_return_vs_initial_pct": round(100 * (net_lp / s.capital_usd - 1), 8),
        "no_position_vs_initial_usd": s.capital_usd,
        "not_included": [
            "V3 price-range behavior", "reward token price movements",
            "slippage and fees beyond entered estimates", "smart-contract risks",
            "depeg and bridging risks", "taxes and withdrawal delays",
        ],
        "warning": (
            "A positive scenario is not an available verified APR or guaranteed "
            "profit; holding the assets can outperform LP. Real loss is possible."
        ),
    }


def prediction_break_even(
    *, estimated_win_probability: float, gross_payout: float,
    treasury_fraction: float = .03, gas_fraction_of_stake: float = 0.0,
) -> dict:
    """What probability would be needed for positive expected bet net-of-fees?

    Gross payout is a PRE-LOCK subjective hypothetical ratio, which is NOT
    guaranteed at settlement; actual pool sizes change before lock.
    All-losing stakes are -100% plus gas. Tie/house-win not modeled.
    """
    vals = (estimated_win_probability, gross_payout, treasury_fraction, gas_fraction_of_stake)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("probability, payout, fee and gas must be finite")
    if not 0 <= estimated_win_probability <= 1:
        raise ValueError("estimated win probability must be in [0,1]")
    if not 0 < treasury_fraction < 1 or gross_payout <= 1 or gas_fraction_of_stake < 0:
        raise ValueError("invalid payout/fee/gas assumptions")
    adjusted = gross_payout * (1 - treasury_fraction)
    needed = (1 + gas_fraction_of_stake) / adjusted
    net = estimated_win_probability * adjusted - 1 - gas_fraction_of_stake
    return {
        "research_only": True,
        "hypothetical_gross_payout": gross_payout,
        "hypothetical_win_probability": estimated_win_probability,
        "break_even_win_probability": needed,
        "hypothetical_expected_net_return_per_stake": net,
        "break_even_feasible": needed <= 1,
        "has_proven_predictive_edge": False,
        "real_pancakeswap_betting_approved": False,
        "critical": (
            "Actual PancakeSwap round payout and oracle settlement are different "
            "from a fixed 2x Binance next-candle scenario. Pool ratios vary until "
            "lock, tie/house-win can lose the whole stake, and gas may be higher."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Offline PancakeSwap V2 LP scenario or Prediction break-even. Never trades.")
    subs = p.add_subparsers(dest="mode", required=True)
    lp = subs.add_parser("lp", help="Constant-product V2 50:50 illustrative LP versus HOLD")
    lp.add_argument("--capital-usd", type=float, required=True)
    lp.add_argument("--token-a-change-pct", type=float, required=True)
    lp.add_argument("--token-b-change-pct", type=float, default=0)
    lp.add_argument("--swap-fee-apr-pct", type=float, default=0)
    lp.add_argument("--farm-apr-pct", type=float, default=0)
    lp.add_argument("--days", type=int, default=30)
    lp.add_argument("--total-gas-usd", type=float, default=0)
    lp.add_argument("--slippage-usd", type=float, default=0)
    bet = subs.add_parser("prediction", help="Hypothetical payout requirements, NOT a betting signal")
    bet.add_argument("--estimated-win-probability", type=float, required=True)
    bet.add_argument("--gross-payout", type=float, required=True)
    bet.add_argument("--treasury-fraction", type=float, default=.03)
    bet.add_argument("--gas-fraction-of-stake", type=float, default=0)
    for command in (lp, bet):
        command.add_argument(
            "--output", type=Path,
            help="Write unique JSON report (refuses overwriting)",
        )
    args = p.parse_args()
    try:
        if args.mode == "lp":
            result = simulate_v2_equal_weight(LpScenario(
                capital_usd=args.capital_usd,
                token_a_change_pct=args.token_a_change_pct,
                token_b_change_pct=args.token_b_change_pct,
                swap_fee_apr_pct=args.swap_fee_apr_pct,
                farm_apr_pct=args.farm_apr_pct,
                days=args.days, total_gas_usd=args.total_gas_usd,
                slippage_usd=args.slippage_usd,
            ))
        else:
            result = prediction_break_even(
                estimated_win_probability=args.estimated_win_probability,
                gross_payout=args.gross_payout,
                treasury_fraction=args.treasury_fraction,
                gas_fraction_of_stake=args.gas_fraction_of_stake,
            )
        if args.output:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite economics report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        p.exit(2, f"Economics aborted: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
