"""Deterministic capital-vs-hold and Prediction economics guardrails."""
import pytest

from flydeck.earn_planner_cli import LpScenario, simulate_v2_equal_weight, prediction_break_even


def test_same_token_prices_illustrative_fees_and_gas():
    r = simulate_v2_equal_weight(LpScenario(
        capital_usd=100, token_a_change_pct=0, token_b_change_pct=0,
        swap_fee_apr_pct=10, farm_apr_pct=0, days=365,
        total_gas_usd=2, slippage_usd=1,
    ))
    assert r["hypothetical_hold_value_usd"] == 100
    assert r["hypothetical_lp_excluding_yield_usd"] == 100
    assert r["illustrative_net_lp_value_usd"] == 107
    assert r["lp_minus_hold_usd"] == 7


def test_imbalanced_prices_impermanent_loss_even_if_dollar_lp_up():
    r = simulate_v2_equal_weight(LpScenario(
        capital_usd=100, token_a_change_pct=300, token_b_change_pct=0,
        days=30,
    ))
    assert r["hypothetical_hold_value_usd"] == 250
    assert r["hypothetical_lp_excluding_yield_usd"] == 200
    assert r["impermanent_loss_relative_to_hold_pct"] == -20
    assert r["lp_minus_hold_usd"] == -50


def test_fees_are_assumptions_not_known_live_yield_and_zero_is_allowed():
    r = simulate_v2_equal_weight(LpScenario(
        capital_usd=60, token_a_change_pct=-50, days=30,
        total_gas_usd=3,
    ))
    assert r["lp_minus_hold_usd"] < 0
    assert r["no_position_vs_initial_usd"] == 60
    assert "V3 price-range behavior" in r["not_included"]


@pytest.mark.parametrize("kwargs", [
    {"capital_usd": 0, "token_a_change_pct": 0},
    {"capital_usd": 100, "token_a_change_pct": -100},
    {"capital_usd": 100, "token_a_change_pct": 0, "farm_apr_pct": -1},
    {"capital_usd": 100, "token_a_change_pct": 0, "total_gas_usd": -1},
    {"capital_usd": 100, "token_a_change_pct": 0, "swap_fee_apr_pct": float("nan")},
])
def test_rejects_invalid_scenarios(kwargs):
    with pytest.raises(ValueError):
        simulate_v2_equal_weight(LpScenario(**kwargs))


def test_prediction_payout_needs_real_edge():
    r = prediction_break_even(estimated_win_probability=.52, gross_payout=2)
    assert r["break_even_win_probability"] == pytest.approx(1 / 1.94)
    assert r["hypothetical_expected_net_return_per_stake"] == pytest.approx(.52 * 1.94 - 1)
    assert not r["real_pancakeswap_betting_approved"]
    assert not r["has_proven_predictive_edge"]


def test_low_payout_undoes_even_60_percent_accuracy():
    r = prediction_break_even(
        estimated_win_probability=.6, gross_payout=1.5,
        gas_fraction_of_stake=.005,
    )
    assert r["hypothetical_expected_net_return_per_stake"] < 0


@pytest.mark.parametrize("kwargs", [
    {"estimated_win_probability": -1, "gross_payout": 2},
    {"estimated_win_probability": 2, "gross_payout": 2},
    {"estimated_win_probability": .5, "gross_payout": 1},
    {"estimated_win_probability": .5, "gross_payout": 2, "treasury_fraction": 1},
    {"estimated_win_probability": .5, "gross_payout": 2, "gas_fraction_of_stake": -1},
])
def test_rejects_invalid_bet_assumptions(kwargs):
    with pytest.raises(ValueError):
        prediction_break_even(**kwargs)
