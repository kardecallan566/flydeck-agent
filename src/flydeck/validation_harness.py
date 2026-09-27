from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable

from .ablation_runner import AblationResult, run_ablation_matrix
from .bnb_prediction_data_runner import BNBPredictionDataset
from .crypto_event_policy import CryptoEventConfig
from .visual_agent import FlyVisualPredictionAgent
from .visual_circuit import VisualCircuit


@dataclass(frozen=True, slots=True)
class FailFastConfig:
    """Controls the cheap gate that must pass before a full ablation."""

    preflight_rounds: int = 1
    max_return_regression: float = 0.02
    max_drawdown_regression: float = 0.05
    min_preflight_rounds: int = 1
    allow_regression: bool = False
    runtime_smoke: bool = False


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    passed: bool
    rounds: int
    gates: tuple[GateResult, ...]
    rows: tuple[AblationResult, ...]


@dataclass(frozen=True, slots=True)
class RunManifest:
    data_path: str
    data_sha256: str
    circuit_path: str
    circuit_sha256: str
    data_size: int
    preflight_rounds: int
    fee_bps: float
    slippage_bps: float
    max_return_regression: float
    max_drawdown_regression: float
    allow_regression: bool
    status: str


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prefix(data: BNBPredictionDataset, rounds: int) -> BNBPredictionDataset:
    size = min(data.size, max(rounds + 1, 32))
    return replace(
        data,
        timestamps=data.timestamps[:size],
        opens=data.opens[:size],
        highs=data.highs[:size],
        lows=data.lows[:size],
        closes=data.closes[:size],
        volumes=data.volumes[:size],
    )


def _row(rows: tuple[AblationResult, ...], variant: str, split: str) -> AblationResult | None:
    return next((item for item in rows if item.variant == variant and item.split == split), None)


def _gate_invariants(rows: tuple[AblationResult, ...]) -> tuple[GateResult, ...]:
    gates: list[GateResult] = []
    expected = {"baseline_original", "malecns_risk", "current_temporal_crypto_event"}
    found = {item.variant for item in rows}
    gates.append(GateResult("variants_present", expected <= found, f"found={sorted(found)}"))
    splits = {item.split for item in rows}
    gates.append(GateResult("splits_present", {"TRAIN", "VALIDATION", "TEST"} <= splits, f"found={sorted(splits)}"))
    finite = True
    for item in rows:
        values = (
            item.standardized.accuracy,
            item.standardized.hit_rate,
            item.standardized.coverage,
            item.standardized.economic_return,
            item.economic.profit_factor,
            item.economic.max_drawdown,
        )
        finite = finite and all(value == value and abs(value) != float("inf") for value in values)
    gates.append(GateResult("finite_metrics", finite, "all metrics finite" if finite else "NaN or infinity detected"))
    return tuple(gates)


def _gate_against_baseline(rows: tuple[AblationResult, ...], config: FailFastConfig) -> GateResult:
    baseline = _row(rows, "baseline_original", "VALIDATION")
    current = _row(rows, "current_temporal_crypto_event", "VALIDATION")
    if baseline is None or current is None:
        return GateResult("no_material_regression", False, "missing baseline/current validation row")
    return_delta = current.economic.net_return - baseline.economic.net_return
    drawdown_delta = current.economic.max_drawdown - baseline.economic.max_drawdown
    passed = (
        return_delta >= -config.max_return_regression
        and drawdown_delta <= config.max_drawdown_regression
    )
    detail = (
        f"return_delta={return_delta:.4%}, drawdown_delta={drawdown_delta:.4%}, "
        f"limits={config.max_return_regression:.4%}/{config.max_drawdown_regression:.4%}"
    )
    return GateResult("no_material_regression", passed, detail)


def run_preflight(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    *,
    config: CryptoEventConfig | None = None,
    fail_fast: FailFastConfig | None = None,
) -> PreflightReport:
    fail_fast = fail_fast or FailFastConfig()
    if data.size < fail_fast.min_preflight_rounds + 1:
        raise ValueError(
            f"dataset has {data.size} candles; at least {fail_fast.min_preflight_rounds + 1} are required"
        )
    rounds = min(fail_fast.preflight_rounds, max(0, data.size - 3))
    rows: tuple[AblationResult, ...] = ()
    gates = [
        GateResult("dataset_loaded", data.size >= fail_fast.min_preflight_rounds + 3, f"candles={data.size}"),
        GateResult("positive_prices", all(value > 0.0 and math.isfinite(value) for value in data.closes[: min(data.size, 64)]), "price prefix valid"),
        GateResult("circuit_loaded", bool(circuit.neurons) and bool(circuit.edges), f"neurons={len(circuit.neurons)}, edges={len(circuit.edges)}"),
    ]
    if fail_fast.runtime_smoke:
        agent = FlyVisualPredictionAgent(circuit, retina_width=32)
        finite = True
        successful = 0
        for index in range(3, rounds + 3):
            prices = data.closes[max(0, index - 31): index + 1]
            volumes = data.volumes[max(0, index - 31): index + 1]
            _stimulus, decision = agent.perceive(prices, volumes=volumes)
            values = (decision.p_up, decision.p_down, decision.p_wait,
                      decision.up_score, decision.down_score, decision.confidence)
            finite = finite and all(math.isfinite(float(value)) for value in values)
            successful += 1
        gates.extend((
            GateResult("runtime_smoke_rounds", successful == rounds, f"completed={successful}/{rounds}"),
            GateResult("finite_decisions", finite, "all decision outputs finite" if finite else "NaN or infinity detected"),
        ))
    passed = all(gate.passed for gate in gates)
    return PreflightReport(passed=passed, rounds=rounds, gates=tuple(gates), rows=rows)


def run_safe_ablation(
    data: BNBPredictionDataset,
    circuit: VisualCircuit,
    *,
    data_path: str | Path,
    circuit_path: str | Path,
    config: CryptoEventConfig | None = None,
    fail_fast: FailFastConfig | None = None,
    manifest_path: str | Path | None = None,
    on_result: Callable[[AblationResult], None] | None = None,
    run_full: bool = True,
) -> tuple[PreflightReport, tuple[AblationResult, ...]]:
    """Run the cheap gate first; only then run the full ablation matrix."""
    config = config or CryptoEventConfig()
    fail_fast = fail_fast or FailFastConfig()
    if manifest_path:
        write_manifest(
            manifest_path,
            RunManifest(
                data_path=str(data_path), data_sha256=sha256_file(data_path),
                circuit_path=str(circuit_path), circuit_sha256=sha256_file(circuit_path),
                data_size=data.size, preflight_rounds=fail_fast.preflight_rounds,
                fee_bps=config.fee_bps, slippage_bps=config.slippage_bps,
                max_return_regression=fail_fast.max_return_regression,
                max_drawdown_regression=fail_fast.max_drawdown_regression,
                allow_regression=fail_fast.allow_regression, status="PREFLIGHT_RUNNING",
            ),
        )
    preflight = run_preflight(data, circuit, config=config, fail_fast=fail_fast)
    if not preflight.passed:
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest["status"] = "PREFLIGHT_FAILED"
            Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return preflight, ()

    if not run_full:
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest["status"] = "PREFLIGHT_PASSED"
            Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return preflight, ()

    full_rows = run_ablation_matrix(data, circuit, config=config, on_result=on_result)
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest["status"] = "COMPLETED"
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preflight, full_rows
