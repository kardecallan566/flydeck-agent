from pathlib import Path

from flydeck.bnb_prediction_data_runner import BNBPredictionDataset
from flydeck.validation_harness import FailFastConfig, sha256_file, write_manifest, RunManifest


def _data(size: int = 20) -> BNBPredictionDataset:
    closes = tuple(100.0 + index * 0.1 for index in range(size))
    return BNBPredictionDataset(
        timestamps=tuple(range(size)),
        opens=closes,
        highs=tuple(value + 0.2 for value in closes),
        lows=tuple(value - 0.2 for value in closes),
        closes=closes,
        volumes=tuple(1.0 for _ in closes),
    )


def test_fail_fast_defaults_are_conservative() -> None:
    config = FailFastConfig()
    assert config.preflight_rounds == 1
    assert config.allow_regression is False
    assert config.runtime_smoke is False
    assert config.max_return_regression < 0.05


def test_sha256_and_manifest_are_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("close\n1\n2\n", encoding="utf-8")
    manifest_path = tmp_path / "run" / "manifest.json"
    write_manifest(
        manifest_path,
        RunManifest(
            data_path=str(source), data_sha256=sha256_file(source),
            circuit_path="circuit.json", circuit_sha256="abc",
            data_size=2, preflight_rounds=2_000, fee_bps=5.0, slippage_bps=2.0,
            max_return_regression=0.02, max_drawdown_regression=0.05,
            allow_regression=False, status="PREFLIGHT_RUNNING",
        ),
    )
    text = manifest_path.read_text(encoding="utf-8")
    assert manifest_path.exists()
    assert '"status": "PREFLIGHT_RUNNING"' in text
    assert len(sha256_file(source)) == 64


def test_prefix_data_contract_is_sized_for_future_outcomes() -> None:
    data = _data()
    assert data.size == 20
    assert data.size > 1
    assert data.outcome(0).name == "UP"
