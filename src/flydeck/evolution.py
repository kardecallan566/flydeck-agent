"""Resource-bounded, chronological population evolution for FlyDeck.

All policies share one causal feature matrix and optionally one cached MaleCNS
signal. Every candidate has its own SGD readout, hyperparameters and lineage.
Research only: this module never submits PancakeSwap transactions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .bnb_prediction import Prediction
from .bnb_prediction_data_runner import BNBPredictionDataset
from .pancakeswap_targets import PancakeAlignment


# Bump whenever neural inference/cache semantics change. V3 disables ALL
# reward-learning submodules, not just the parent boolean.
FLY_CACHE_VERSION = "frozen-inference-v3-set-learning"

FEATURE_NAMES = (
    "bias", "return_1", "return_2", "return_3", "return_6",
    "return_12", "return_24", "volatility_6", "volatility_24",
    "volume_surprise", "candle_body", "close_location", "fly_signal",
)
FAMILIES = (
    "short", "trend", "reversion", "volume", "volatility",
    "candle", "fly", "mixed", "selective", "exploratory",
)


def market_features(data: BNBPredictionDataset, fly_signal: np.ndarray | None = None) -> np.ndarray:
    """Compute all causal features ONCE, for all agents."""
    n = data.size
    close = np.asarray(data.closes, dtype=np.float64)
    open_ = np.asarray(data.opens, dtype=np.float64)
    high = np.asarray(data.highs, dtype=np.float64)
    low = np.asarray(data.lows, dtype=np.float64)
    volume = np.asarray(data.volumes, dtype=np.float64)
    if n < 35 or np.any(close <= 0) or not np.all(np.isfinite(close)):
        raise ValueError("need >=35 finite positive price candles")
    values = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    values[:, 0] = 1.0
    returns = {}
    for col, lag in zip(range(1, 7), (1, 2, 3, 6, 12, 24)):
        ret = np.zeros(n, dtype=np.float64)
        ret[lag:] = close[lag:] / close[:-lag] - 1.0
        returns[lag] = ret
        values[:, col] = np.clip(ret * 250, -3, 3)
    one = returns[1]
    for col, width in ((7, 6), (8, 24)):
        cs = np.concatenate(([0.0], np.cumsum(one)))
        cs2 = np.concatenate(([0.0], np.cumsum(one * one)))
        index = np.arange(n)
        start = np.maximum(0, index + 1 - width)
        count = index + 1 - start
        mean = (cs[index + 1] - cs[start]) / count
        variance = (cs2[index + 1] - cs2[start]) / count - mean * mean
        values[:, col] = np.clip(np.sqrt(np.maximum(0.0, variance)) * 250, 0, 3)
    csvol = np.concatenate(([0.0], np.cumsum(volume)))
    ix = np.arange(n)
    start = np.maximum(0, ix - 5)
    baseline = (csvol[ix] - csvol[start]) / np.maximum(1, ix - start)
    values[:, 9] = np.clip(volume / np.maximum(baseline, 1e-9) - 1, -3, 3)
    values[:2, 9] = 0
    values[:, 10] = np.clip((close - open_) / np.maximum(open_, 1e-9) * 250, -3, 3)
    values[:, 11] = np.clip(2 * (close - low) / np.maximum(high - low, 1e-9) - 1, -1, 1)
    if fly_signal is None:
        values[:, 12] = 0
    else:
        fly = np.asarray(fly_signal, dtype=np.float32)
        if fly.shape != (n,) or not np.all(np.isfinite(fly)):
            raise ValueError("fly signal must be a finite vector matching dataset size")
        values[:, 12] = np.clip(fly, -1, 1)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite features")
    values.flags.writeable = False
    return values


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fly_feature_cache(
    data: BNBPredictionDataset, *, market_file: Path, circuit_file: Path,
    cache_file: Path, context: int = 32, rebuild: bool = False,
) -> np.ndarray:
    """Run ONE frozen shared visual circuit, cache by both input-file hashes."""
    fingerprint = ":".join((FLY_CACHE_VERSION, _hash(market_file), _hash(circuit_file), str(context)))
    if cache_file.exists() and not rebuild:
        with np.load(cache_file, allow_pickle=False) as loaded:
            if str(loaded["fingerprint"].item()) != fingerprint:
                raise ValueError("Fly cache is stale (inference/version/input changed): supply --rebuild-fly-cache")
            signal = loaded["signal"].astype(np.float32)
            timestamps = loaded["timestamps"].astype(np.int64)
        if signal.shape != (data.size,) or not np.array_equal(timestamps, data.timestamps):
            raise ValueError("Fly cache timestamps do not match historical CSV")
        return signal

    from .visual_agent import FlyVisualPredictionAgent
    from .visual_circuit import VisualCircuit

    circuit = VisualCircuit.load(circuit_file)
    agent = FlyVisualPredictionAgent(circuit, confidence_threshold=0.0)
    agent.set_learning(False)  # Propagate to decision engine and episodic memory, too.
    signal = np.zeros(data.size, dtype=np.float32)
    print(f"Building shared MaleCNS cache: {data.size} candles, {agent.neuron_count} neurons")
    for i in range(max(32, context) - 1, data.size):
        start = i - context + 1
        _, decision = agent.perceive(
            data.closes[start:i + 1], volumes=data.volumes[start:i + 1],
        )
        signal[i] = np.clip(decision.p_up - decision.p_down, -1, 1)
        if (i + 1) % 2000 == 0:
            print(f"  shared fly features: {i + 1}/{data.size}", flush=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache_file.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
        np.savez_compressed(
            temp, signal=signal, timestamps=np.asarray(data.timestamps, dtype=np.int64),
            fingerprint=np.array(fingerprint),
        )
    os.replace(temporary, cache_file)
    return signal


@dataclass(slots=True)
class Candidate:
    agent_id: str
    family: str
    parent_id: str
    generation: int
    weights: np.ndarray
    mask: np.ndarray
    learning_rate: float
    l2: float
    threshold: float


@dataclass(frozen=True, slots=True)
class WindowResult:
    stage: str
    block: int
    agent_id: str
    family: str
    parent_id: str
    generation: int
    eligible: int
    entered: int
    correct: int
    up: int
    down: int
    accuracy: float
    coverage: float
    equity: float
    max_drawdown: float
    survived: bool  # Profitable AND within participation / drawdown constraints
    fitness: float
    risk_survived: bool = False
    profitable: bool = False
    wilson_lower_95: float = 0.0
    break_even_probability: float | None = None
    statistical_evidence: bool = False


@dataclass(frozen=True, slots=True)
class EvolutionSettings:
    population: int = 100
    seed: int = 42
    block_size: int = 10_000
    warmup_blocks: int = 3
    development_blocks: int = 2
    validation_blocks: int = 1
    audit_blocks: int = 1
    survivors: float = 0.20
    min_entries: int = 80
    min_coverage: float = 0.05
    stake_fraction: float = 0.002
    max_drawdown: float = 0.25
    scenario_gross_odds: float = 2.0
    scenario_fee: float = 0.03
    gas_fraction_of_stake: float = 0.0
    finalists: int = 5

    def __post_init__(self) -> None:
        if not (10 <= self.population <= 300 and self.population % 10 == 0):
            raise ValueError("population must be a multiple of 10, from 10 to 300")
        if not (0 < self.survivors <= 0.5 and self.finalists <= self.population):
            raise ValueError("invalid survivor/finalist count")
        if not (0 < self.stake_fraction <= 0.01 and 0 < self.max_drawdown < 1):
            raise ValueError("invalid risk limits")
        if not (0 <= self.scenario_fee < 1 and self.scenario_gross_odds > 1):
            raise ValueError("invalid payout assumptions")
        if min(self.block_size, self.min_entries, self.warmup_blocks,
               self.development_blocks, self.validation_blocks, self.audit_blocks) < 1:
            raise ValueError("block sizes and phase counts must be positive")


def _family_mask(family: str, fly_available: bool) -> np.ndarray:
    mask = np.full(len(FEATURE_NAMES), 0.35)
    groups = {
        "short": (1, 2, 3, 10),
        "trend": (4, 5, 6),
        "reversion": (1, 4, 11),
        "volume": (1, 9, 10),
        "volatility": (7, 8, 11),
        "candle": (9, 10, 11),
        "fly": (12, 1, 8),
        "mixed": tuple(range(1, 13)),
        "selective": (1, 6, 8, 12),
        "exploratory": tuple(range(1, 13)),
    }
    mask[list(groups[family])] = 1.0
    if not fly_available:
        mask[12] = 0.0
    mask[0] = 1.0
    return mask


def initial_population(settings: EvolutionSettings, *, fly_available: bool) -> list[Candidate]:
    rng = np.random.default_rng(settings.seed)
    agents = []
    for i in range(settings.population):
        family = FAMILIES[i % len(FAMILIES)]
        agents.append(Candidate(
            agent_id=f"A{i + 1:04d}", family=family, parent_id="",
            generation=0, weights=rng.normal(0, 0.07, len(FEATURE_NAMES)),
            mask=_family_mask(family, fly_available),
            learning_rate=float(rng.uniform(0.001, 0.015)),
            l2=float(rng.uniform(0.00001, 0.001)),
            threshold=float(rng.uniform(0.05, 0.40)),
        ))
    return agents


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))


def _wilson_lower(correct: int, entered: int, z: float = 1.95996398454) -> float:
    """Approximate binomial lower confidence bound; not a proof of trading edge."""
    if entered <= 0:
        return 0.0
    p = correct / entered
    z2 = z * z
    return max(0.0, (p + z2 / (2 * entered)
        - z * math.sqrt((p * (1 - p) + z2 / (4 * entered)) / entered)
    ) / (1 + z2 / entered))


def _labels(
    data: BNBPredictionDataset, alignment: PancakeAlignment | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    y = np.full(data.size, -1, dtype=np.int8)
    ready = np.arange(data.size, dtype=np.int64) + 1
    epochs: dict[int, int] = {}
    if alignment:
        # Earliest point with a fully closed observation after the oracle close.
        import bisect
        for row in alignment.aligned_rounds:
            idx = row.feature_index
            y[idx] = 1 if row.outcome == Prediction.UP else 0
            ready[idx] = bisect.bisect_left(data.timestamps, row.close_timestamp_ms) + 1
            epochs[idx] = row.epoch
    else:
        c = np.asarray(data.closes, dtype=np.float64)
        y[:-1] = np.where(c[1:] > c[:-1], 1, np.where(c[1:] < c[:-1], 0, -1))
    return y, ready, epochs


def load_odds_snapshots(
    csv_path: Path, alignment: PancakeAlignment, *,
    max_age_ms: int = 120_000,
) -> dict[int, tuple[float, float]]:
    """Pre-lock quotes only; NEVER reconstruct entry odds from final pool totals.

    CSV: epoch,timestamp_ms,gross_up,gross_down. Gross = payout before treasury
    fee, NOT the stake profit. Missing/stale quotes are excluded from P&L.
    """
    snapshots: dict[int, tuple[int, float, float]] = {}
    cutoff_by_epoch = {r.epoch: r.decision_timestamp_ms for r in alignment.aligned_rounds}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch not in cutoff_by_epoch:
                continue
            ts = int(row["timestamp_ms"])
            if ts < 10**12:
                ts *= 1000
            age = cutoff_by_epoch[epoch] - ts
            if not 0 <= age <= max_age_ms:
                continue
            odds = (float(row["gross_up"]), float(row["gross_down"]))
            if not all(math.isfinite(o) and o > 1 for o in odds):
                continue
            if epoch not in snapshots or snapshots[epoch][0] < ts:
                snapshots[epoch] = (ts, *odds)
    return {epoch: (row[1], row[2]) for epoch, row in snapshots.items()}


def _block(
    agents: list[Candidate], x: np.ndarray, y: np.ndarray, ready: np.ndarray,
    start: int, stop: int, *, stage: str, block: int,
    settings: EvolutionSettings, learn: bool, odds: dict[int, tuple[float, float]] | None,
    epochs: dict[int, int],
    trace_rows: list[dict] | None = None,
    trace_agents: set[str] | None = None,
    timestamp_ms: tuple[int, ...] | None = None,
    initial_state: dict[str, dict] | None = None,
    state_out: dict[str, dict] | None = None,
) -> list[WindowResult]:
    """Vectorized policy update; online feedback is delayed until resolution."""
    count = len(agents)
    weights = np.stack([a.weights for a in agents])
    masks = np.stack([a.mask for a in agents])
    lr = np.asarray([a.learning_rate for a in agents])
    l2 = np.asarray([a.l2 for a in agents])
    threshold = np.asarray([a.threshold for a in agents])
    bank = np.full(count, 100.0)
    peak = bank.copy()
    drawdown = np.zeros(count)
    entered = np.zeros(count, dtype=np.int64)
    correct = np.zeros(count, dtype=np.int64)
    up = np.zeros(count, dtype=np.int64)
    down = np.zeros(count, dtype=np.int64)
    halted = np.zeros(count, dtype=bool)
    if initial_state is not None:
        for j, agent in enumerate(agents):
            prev = initial_state.get(agent.agent_id)
            if prev is None:
                continue
            bank[j] = float(prev["equity"])
            peak[j] = float(prev["peak"])
            drawdown[j] = float(prev["max_drawdown"])
            halted[j] = bool(prev["halted"])
    eligible = 0
    pending: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    odds_default = (settings.scenario_gross_odds, settings.scenario_gross_odds)
    for i in range(start, stop):
        # All queued labels become learnable only after their resolution time.
        if learn and pending:
            due = [event for event in pending if event[0] <= i]
            pending = [event for event in pending if event[0] > i]
            for _, feature, probability, target in due:
                gradient = (target - probability)[:, None] * feature[None, :] * masks
                weights += lr[:, None] * (gradient - l2[:, None] * weights)
                np.clip(weights, -3, 3, out=weights)

        if y[i] < 0 or ready[i] >= stop:
            continue
        odds_here = odds_default
        if odds is not None:
            odds_here = odds.get(epochs.get(i, -1))
            if odds_here is None:
                # Directional learning remains valid; economics are unavailable.
                if learn:
                    f = x[i].astype(np.float64, copy=False)
                    prob = _sigmoid(np.sum(weights * (masks * f), axis=1))
                    pending.append((int(ready[i]), f, prob.copy(), float(y[i])))
                continue

        eligible += 1
        f = x[i].astype(np.float64, copy=False)
        prob = _sigmoid(np.sum(weights * (masks * f), axis=1))
        action_up = prob >= 0.5
        confidence = 2.0 * np.abs(prob - 0.5)
        chosen = (confidence >= threshold) & ~halted
        entered += chosen
        up += chosen & action_up
        down += chosen & ~action_up
        success = action_up == bool(y[i])
        correct += chosen & success
        gross = np.where(action_up, odds_here[0], odds_here[1])
        net_unit = np.where(
            success, gross * (1.0 - settings.scenario_fee) - 1.0, -1.0,
        ) - settings.gas_fraction_of_stake
        bank_before = bank.copy() if trace_rows is not None else None
        bank += bank * settings.stake_fraction * net_unit * chosen
        peak = np.maximum(peak, bank)
        drawdown = np.maximum(drawdown, (peak - bank) / np.maximum(peak, 1e-9))
        halted |= drawdown >= settings.max_drawdown
        if trace_rows is not None:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is mandatory when tracing predictions")
            if trace_agents is None:
                raise ValueError("trace_agents is mandatory to bound memory")
            for j, agent in enumerate(agents):
                if agent.agent_id not in trace_agents:
                    continue
                action = ("UP" if bool(action_up[j]) else "DOWN") if bool(chosen[j]) else "WAIT"
                trace_rows.append({
                    "stage": stage, "block": block,
                    "agent_id": agent.agent_id, "family": agent.family,
                    "feature_index": i,
                    "feature_candle_open_ms": int(timestamp_ms[i]),
                    "decision_earliest_ms": int(timestamp_ms[i]) + 300_000,
                    "outcome_available_index": int(ready[i]),
                    "epoch": epochs.get(i, ""),
                    "p_up": float(prob[j]), "confidence": float(confidence[j]),
                    "threshold": float(threshold[j]), "action": action,
                    "wait_reason": (
                        "" if bool(chosen[j]) else
                        "risk_halt" if bool(halted[j]) else "low_confidence"
                    ),
                    "resolved_outcome": "UP" if y[i] == 1 else "DOWN",
                    "correct": (bool(success[j]) if bool(chosen[j]) else ""),
                    "quote_gross": (float(gross[j]) if bool(chosen[j]) else ""),
                    "equity_before": float(bank_before[j]),
                    "equity_after": float(bank[j]),
                    "economic_mode": "prelock_snapshot" if odds is not None else "illustrative_scenario",
                })
        if learn:
            pending.append((int(ready[i]), f, prob.copy(), float(y[i])))

    if learn:
        # Labels still pending at block end are discarded rather than leaking
        # into an earlier generation or across purge boundaries.
        for j, agent in enumerate(agents):
            agent.weights = weights[j].copy()

    if state_out is not None:
        for j, agent in enumerate(agents):
            state_out[agent.agent_id] = {
                "equity": float(bank[j]), "peak": float(peak[j]),
                "max_drawdown": float(drawdown[j]), "halted": bool(halted[j]),
            }

    results = []
    for j, agent in enumerate(agents):
        entries = int(entered[j])
        coverage = entries / eligible if eligible else 0.0
        risk_survived = (
            not halted[j] and entries >= settings.min_entries
            and coverage >= settings.min_coverage
        )
        profitable = float(bank[j]) > 100.0 + 1e-9
        # Economic scenario is an explicit assumption. A variable pre-lock
        # payout requires a per-bet analysis, not a single break-even threshold.
        break_even = (
            (1 + settings.gas_fraction_of_stake)
            / (settings.scenario_gross_odds * (1 - settings.scenario_fee))
            if odds is None else None
        )
        wilson = _wilson_lower(int(correct[j]), entries)
        evidence = bool(
            risk_survived and profitable and break_even is not None
            and wilson > break_even
        )
        survived = bool(risk_survived and profitable)
        # Risk-qualified but losing policies can remain as diverse exploratory
        # parents. They are NEVER marked as profitable or as surviving profit.
        fitness = (
            math.log(max(1e-9, float(bank[j]) / 100.0))
            - 2 * float(drawdown[j]) + 0.02 * coverage
            if risk_survived else -1e6 + entries / max(1, eligible)
        )
        results.append(WindowResult(
            stage, block, agent.agent_id, agent.family, agent.parent_id,
            agent.generation, eligible, entries, int(correct[j]), int(up[j]),
            int(down[j]), int(correct[j]) / entries if entries else 0.0,
            coverage, float(bank[j]), float(drawdown[j]), survived, fitness,
            bool(risk_survived), bool(profitable), wilson, break_even, evidence,
        ))
    return results


def evolve(
    agents: list[Candidate], results: list[WindowResult], *,
    settings: EvolutionSettings, generation: int, fly_available: bool,
) -> list[Candidate]:
    """Keep elites and diversity representatives; clone/mutate the rest."""
    rng = np.random.default_rng(settings.seed + 7919 * generation)
    by_id = {r.agent_id: r for r in results}
    ranked = sorted(agents, key=lambda a: by_id[a.agent_id].fitness, reverse=True)
    keep = max(len(FAMILIES), int(settings.population * settings.survivors))
    # Guarantee one representative per family, then fill remaining slots
    # by fitness. Compare IDs, not dataclasses with NumPy arrays.
    unique: list[Candidate] = []
    seen: set[str] = set()
    for family in FAMILIES:
        representative = next((a for a in ranked if a.family == family), None)
        if representative is not None:
            unique.append(representative)
            seen.add(representative.agent_id)
    for agent in ranked:
        if len(unique) >= keep:
            break
        if agent.agent_id not in seen:
            unique.append(agent)
            seen.add(agent.agent_id)
    next_generation = list(unique)
    for i in range(settings.population - len(unique)):
        parent = unique[i % len(unique)]
        child_id = f"G{generation:02d}-{i + 1:04d}"
        weights = parent.weights.copy() + rng.normal(0, 0.035, len(FEATURE_NAMES))
        next_generation.append(Candidate(
            agent_id=child_id, family=parent.family, parent_id=parent.agent_id,
            generation=generation, weights=np.clip(weights, -3, 3),
            mask=_family_mask(parent.family, fly_available),
            learning_rate=float(np.clip(parent.learning_rate * rng.uniform(0.75, 1.25), 0.0002, 0.04)),
            l2=float(np.clip(parent.l2 * rng.uniform(0.5, 1.5), 0.000001, 0.01)),
            threshold=float(np.clip(parent.threshold + rng.normal(0, 0.045), 0.02, 0.8)),
        ))
    return next_generation


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _agent_details(agent: Candidate) -> dict:
    return {
        "agent_id": agent.agent_id, "family": agent.family,
        "parent_id": agent.parent_id, "generation": agent.generation,
        "learning_rate": agent.learning_rate, "l2": agent.l2,
        "threshold": agent.threshold, "weights": [float(w) for w in agent.weights],
    }


def run_evolution(
    data: BNBPredictionDataset, settings: EvolutionSettings, *,
    output: Path, fly_signal: np.ndarray | None = None,
    alignment: PancakeAlignment | None = None,
    odds_by_epoch: dict[int, tuple[float, float]] | None = None,
    recent: BNBPredictionDataset | None = None,
    recent_fly_signal: np.ndarray | None = None,
    recent_alignment: PancakeAlignment | None = None,
    recent_odds_by_epoch: dict[int, tuple[float, float]] | None = None,
    recent_holdout: int = 2016,
) -> dict:
    """All populations produce individual CSV results; no real-money execution."""
    required_blocks = (
        settings.warmup_blocks + settings.development_blocks
        + settings.validation_blocks + settings.audit_blocks
    )
    if data.size < required_blocks * settings.block_size - 1:
        raise ValueError(f"need approximately {required_blocks * settings.block_size} historical candles")
    if alignment is not None and data.target_name == "binance-close-t+1":
        raise ValueError("alignment dataset must be used with official outcomes")
    features = market_features(data, fly_signal)
    y, ready, epochs = _labels(data, alignment)
    agents = initial_population(settings, fly_available=fly_signal is not None)
    lineage: dict[str, dict] = {a.agent_id: _agent_details(a) for a in agents}
    history: list[dict] = []
    block_results: dict[str, list[WindowResult]] = {}
    phases = (
        ["train"] * settings.warmup_blocks
        + ["development"] * settings.development_blocks
        + ["validation"] * settings.validation_blocks
        + ["historical_audit"] * settings.audit_blocks
    )
    finalists: list[str] = []
    audit_trace: list[dict] = []
    for b, stage in enumerate(phases):
        begin = 31 + b * settings.block_size if b == 0 else b * settings.block_size
        end = min(data.size - 1, (b + 1) * settings.block_size)
        if end <= begin:
            raise ValueError("empty chronological block")
        learn = stage in {"train", "development"}
        results = _block(
            agents, features, y, ready, begin, end, stage=stage, block=b,
            settings=settings, learn=learn, odds=odds_by_epoch, epochs=epochs,
            trace_rows=audit_trace if stage == "historical_audit" else None,
            trace_agents=set(finalists) if stage == "historical_audit" else None,
            timestamp_ms=data.timestamps if stage == "historical_audit" else None,
        )
        history.extend(asdict(r) for r in results)
        block_results[f"{stage}-{b}"] = results
        best = sorted(results, key=lambda r: r.fitness, reverse=True)[:3]
        print(f"Block {b + 1}/{len(phases)} [{stage}] "
              f"eligible={results[0].eligible} "
              + " | ".join(f"{r.agent_id} {r.accuracy:.1%} eq={r.equity:.2f}"
                           for r in best), flush=True)
        if stage == "validation":
            # Chosen ONLY on historical validation. Audit and recent do not
            # rewrite final status or the candidate shortlist.
            finalists = [r.agent_id for r in sorted(
                results, key=lambda r: r.fitness, reverse=True
            ) if r.survived][:settings.finalists]
        if learn:
            agents = evolve(
                agents, results, settings=settings, generation=b + 1,
                fly_available=fly_signal is not None,
            )
            for agent in agents:
                lineage.setdefault(agent.agent_id, _agent_details(agent))

    validation = next(r for k, r in block_results.items() if k.startswith("validation"))
    audit = next(r for k, r in block_results.items() if k.startswith("historical_audit"))
    by_val = {r.agent_id: r for r in validation}
    by_audit = {r.agent_id: r for r in audit}
    final_rows = []
    for agent in agents:
        v, a = by_val[agent.agent_id], by_audit[agent.agent_id]
        final_rows.append({
            **{k: w for k, w in _agent_details(agent).items() if k != "weights"},
            "validation_accuracy": v.accuracy, "validation_coverage": v.coverage,
            "validation_equity": v.equity, "validation_max_drawdown": v.max_drawdown,
            "validation_survived": v.survived,
            "validation_risk_survived": v.risk_survived,
            "validation_profitable": v.profitable,
            "validation_statistical_evidence": v.statistical_evidence,
            "audit_accuracy": a.accuracy, "audit_coverage": a.coverage,
            "audit_equity": a.equity, "audit_max_drawdown": a.max_drawdown,
            "audit_survived": a.survived,
            "audit_risk_survived": a.risk_survived,
            "audit_profitable": a.profitable,
            "audit_statistical_evidence": a.statistical_evidence,
            "positive_both_validation_and_audit": bool(v.profitable and a.profitable),
            "finalist": agent.agent_id in finalists,
        })
    final_rows.sort(key=lambda row: by_val[row["agent_id"]].fitness, reverse=True)
    _save_csv(output / "all_block_results.csv", history)
    _save_csv(output / "historical_audit_finalist_decisions.csv", audit_trace)
    _save_csv(output / "final_population.csv", final_rows)
    _save_csv(output / "lineage.csv", [
        {k: v for k, v in details.items() if k != "weights"}
        for details in lineage.values()
    ])
    summary = {
        "research_only": True, "real_execution_enabled": False,
        "population": settings.population, "seed": settings.seed,
        "historical_target": data.target_name,
        "economic_mode": "prelock_snapshot" if odds_by_epoch is not None else "illustrative_scenario_not_actual_pnl",
        "fly_features": "shared_cached_malecns" if fly_signal is not None else "OHLCV_only_screening",
        "settings": asdict(settings), "finalists_selected_before_audit_or_recent": finalists,
        "qualification": {
            "risk_survived": "entries, coverage, max drawdown",
            "profitable": "equity above initial 100 in this one block",
            "survived": "risk_survived AND profitable",
            "statistical_evidence": "illustrative Wilson lower bound exceeds fixed-scenario break even; NOT trading approval",
            "validation_survived_count": sum(r.survived for r in validation),
            "audit_survived_count": sum(r.survived for r in audit),
            "positive_both_blocks_count": sum(
                by_val[a.agent_id].profitable and by_audit[a.agent_id].profitable
                for a in agents
            ),
            "risk_and_profit_both_blocks_count": sum(
                by_val[a.agent_id].survived and by_audit[a.agent_id].survived
                for a in agents
            ),
            "historical_qualified_but_not_live_approved": True,
        },
        "note": "Historical audit was previously examined; do not treat it as a fresh independent test.",
        "recent_test": "not_supplied",
        "male_cns_cache_version": FLY_CACHE_VERSION if fly_signal is not None else None,
    }

    # Save the FROZEN, auditable model state BEFORE any optional recent adaptation.
    # Checkpoint all 100+ agents so that candidate selection can be declared
    # before the NEXT, never-before-seen prospective test.
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "source_target": data.target_name,
        "frozen_before_recent": True, "research_only": True,
        "last_historical_candle_open_ms": int(data.timestamps[-1]),
        "population": settings.population,
        "feature_names": list(FEATURE_NAMES),
        "has_shared_fly_signal": fly_signal is not None,
        "fly_cache_version": FLY_CACHE_VERSION if fly_signal is not None else None,
        "finalists_selected_on_validation": finalists,
        "settings": asdict(settings),
        "agents": [
            {**_agent_details(a), "mask": [float(v) for v in a.mask]}
            for a in agents
        ],
    }
    (output / "population_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    summary["checkpoint_before_recent"] = "population_checkpoint.json"

    if recent is not None:
        # All finalists and non-finalists share the same unseen holdout;
        # nobody is selected/promoted based on this holdout.
        if recent_holdout < 100 or recent_holdout > recent.size or recent.size < 35:
            raise ValueError("recent dataset too small or invalid holdout")
        if recent.timestamps[0] <= data.timestamps[-1]:
            raise ValueError("recent dataset overlaps historical timestamps; provide strictly newer candles")
        rx = market_features(recent, recent_fly_signal)
        ry, rr, re = _labels(recent, recent_alignment)
        end_train = recent.size - recent_holdout
        if end_train > 33:
            _block(
                agents, rx, ry, rr, 32, end_train - 1, stage="recent_adaptation",
                block=0, settings=settings, learn=True, odds=recent_odds_by_epoch,
                epochs=re,
            )
        recent_trace: list[dict] = []
        recent_rows = _block(
            agents, rx, ry, rr, max(32, end_train), recent.size - 1,
            stage="recent_sealed", block=1, settings=settings, learn=False,
            odds=recent_odds_by_epoch, epochs=re,
            trace_rows=recent_trace, trace_agents=set(finalists),
            timestamp_ms=recent.timestamps,
        )
        _save_csv(output / "recent_sealed_finalist_decisions.csv", recent_trace)
        _save_csv(output / "recent_sealed_results.csv", [
            asdict(r) for r in recent_rows
        ])
        recent_by_id = {r.agent_id: r for r in recent_rows}
        summary["recent_test"] = {
            "first_timestamp_ms": int(recent.timestamps[max(32, end_train)]),
            "warmup_skipped": max(32 - end_train, 0),
            "adaptation_candles": max(0, end_train),
            "sealed_candles_requested": recent_holdout,
            "sealed_candles_evaluable": max(0, recent.size - 1 - max(32, end_train)),
            "finalist_results": {
                identifier: asdict(recent_by_id[identifier]) for identifier in finalists
            },
            "last_timestamp_ms": int(recent.timestamps[-1]),
            "target": recent.target_name,
            "entries_evaluated_per_agent": {
                r.agent_id: r.entered for r in recent_rows
            },
            "no_promotions_from_sealed_test": True,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (output / "finalists.json").write_text(
        json.dumps([
            {**_agent_details(a), "validation": asdict(by_val[a.agent_id]),
             "historical_audit": asdict(by_audit[a.agent_id])}
            for a in agents if a.agent_id in finalists
        ], indent=2) + "\n", encoding="utf-8",
    )
    return summary
