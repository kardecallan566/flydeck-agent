"""V2-only 50/50 constant-product stress comparison versus holding both assets.

User-specified deterministic HYPOTHESES, NEVER forecasts or quotes. V3 ranges
are intentionally excluded: assuming V2 dynamics would be misleading.
"""
from __future__ import annotations

import math


def v2_stress_scenarios(
    *, capital_usd: float, hypothetical_gross_yield_usd: float,
    roundtrip_cost_usd: float, shock_pct: float = 30.0,
) -> dict:
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool)
        and math.isfinite(x)
        for x in (capital_usd, hypothetical_gross_yield_usd,
                  roundtrip_cost_usd, shock_pct)
    ):
        raise ValueError("V2 stress scenario inputs must be finite numbers")
    if capital_usd <= 0 or hypothetical_gross_yield_usd < 0 or roundtrip_cost_usd < 0:
        raise ValueError("capital must be positive; gross yield and costs non-negative")
    if not 1 <= shock_pct <= 90:
        raise ValueError("stress shock must be between 1% and 90%")
    changes = sorted(set((-shock_pct, -10.0, 0.0, 10.0, shock_pct)))
    rows = []
    for change in changes:
        relative_price = 1 + change / 100
        # Initial V2 50/50 position, token B USD price fixed ONLY in this
        # hypothetical scenario; fee revenue is an UNVERIFIED simple USD input.
        hodl = capital_usd * (1 + relative_price) / 2
        lp_before_yield = capital_usd * math.sqrt(relative_price)
        impermanent_loss_pct = 100 * (lp_before_yield / hodl - 1)
        lp_after = lp_before_yield + hypothetical_gross_yield_usd - roundtrip_cost_usd
        rows.append({
            "token_a_price_change_pct": change,
            "token_b_price_change_pct": 0.0,
            "hold_50_50_usd": round(hodl, 6),
            "lp_without_rewards_usd": round(lp_before_yield, 6),
            "impermanent_loss_vs_hold_pct": round(impermanent_loss_pct, 6),
            "net_lp_usd": round(lp_after, 6),
            "lp_minus_hold_usd": round(lp_after - hodl, 6),
            "net_lp_vs_initial_usd": round(lp_after - capital_usd, 6),
        })
    return {
        "model": "V2_CONSTANT_PRODUCT_50_50_ONE_ASSET_MOVES",
        "model_is_not_v3": True,
        "single_asset_changes_are_hypothetical_not_predictions": True,
        "other_asset_assumed_unchanged_for_illustration": True,
        "reported_apy_not_verified_realizable": True,
        "rewards_are_assumed_constant_usd_for_all_scenarios": True,
        "capital_usd": capital_usd,
        "hypothetical_gross_yield_usd": hypothetical_gross_yield_usd,
        "assumed_total_roundtrip_cost_usd": roundtrip_cost_usd,
        "rows": rows,
        "warning": (
            "This is NOT a forecast or a user's realized PancakeSwap PnL. "
            "Only an idealized V2 50/50 pool is modeled. "
            "Either token can change price, pool yields and reward token "
            "USD values can fall, and losses can be far worse than shown."
        ),
    }
