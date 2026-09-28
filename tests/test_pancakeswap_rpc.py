from flydeck.pancakeswap_rpc import decode_bnb_prediction_round


def _word(value: int) -> str:
    if value < 0:
        value = (1 << 256) + value
    return f"{value:064x}"


def test_decode_official_bnb_prediction_v2_round_layout() -> None:
    values = [
        500_000,
        1_700_000_000,
        1_700_000_300,
        1_700_000_600,
        60_000_000_000,
        60_100_000_000,
        111,
        112,
        10**20,
        4 * 10**19,
        6 * 10**19,
        4 * 10**19,
        97 * 10**18,
        1,
    ]
    encoded = "0x" + "".join(_word(value) for value in values)
    row = decode_bnb_prediction_round(encoded)
    assert row.epoch == 500_000
    assert row.lock_timestamp_ms == 1_700_000_300_000
    assert row.close_price > row.lock_price
    assert row.oracle_called is True
    assert row.bull_amount == float(4 * 10**19)


def test_decode_handles_signed_chainlink_prices() -> None:
    values = [1, 1, 2, 3, -10, 20, 1, 2, 0, 0, 0, 0, 0, 1]
    encoded = "0x" + "".join(_word(value) for value in values)
    row = decode_bnb_prediction_round(encoded)
    assert row.lock_price == -10.0
    assert row.close_price == 20.0
