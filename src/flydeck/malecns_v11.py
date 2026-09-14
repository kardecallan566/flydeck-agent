from __future__ import annotations

import random
from dataclasses import dataclass

from .finance import Candle, CryptoTradingEnvironment
from .malecns import MaleCNSEdge, MaleCNSCircuit, MaleCNSNeuron, MaleCNSReservoir


@dataclass(frozen=True, slots=True)
class Checkpoint:
    step: int
    epsilon: float
    scores: tuple[float, float, float]
    td_error_mean: float
    reward_mean: float


@dataclass(frozen=True, slots=True)
class ReservoirDiagnostics:
    mean_activity: float
    activity_std: float
    active_neurons_mean: float
    unique_states: int
    tanh_saturation_pct: float


@dataclass(frozen=True, slots=True)
class ReadoutDiagnostics:
    weights: tuple[float, float, float]
    biases: tuple[float, float, float]
    score_margin_mean: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class V11Diagnostics:
    checkpoints: tuple[Checkpoint, ...]
    reservoir: ReservoirDiagnostics
    readout: ReadoutDiagnostics
    trained_test_actions: tuple[int, int, int]
    random_topology_test_actions: tuple[int, int, int]
    random_topology_edges: int


def build_random_topology(circuit: MaleCNSCircuit, *, seed: int = 123) -> MaleCNSCircuit:
    """Create a topology control with identical neuron/pool sizes and edge count."""
    rng = random.Random(seed)
    count = len(circuit.neurons)
    used: set[tuple[int, int]] = set()
    edges: list[MaleCNSEdge] = []
    target_edges = len(circuit.edges)
    while len(edges) < target_edges:
        source = rng.randrange(count)
        target = rng.randrange(count)
        key = (source, target)
        if key in used:
            continue
        used.add(key)
        weight = rng.uniform(-0.5, 0.5)
        edges.append(MaleCNSEdge(source, target, weight))
    neurons = tuple(MaleCNSNeuron(n.body_id, n.role, n.neurotransmitter) for n in circuit.neurons)
    return MaleCNSCircuit(neurons, tuple(edges), circuit.input_neurons, circuit.output_neurons)


def run_v11_diagnostics(
    circuit: MaleCNSCircuit,
    candles: tuple[Candle, ...],
    *,
    seed: int = 42,
    checkpoint_interval: int = 5000,
    train_epsilon: float = 0.20,
    train_epsilon_decay: float = 0.9999,
    min_epsilon: float = 0.05,
    learning_rate: float = 0.005,
    discount: float = 0.97,
) -> V11Diagnostics:
    reservoir = MaleCNSReservoir(circuit, feature_count=12, action_count=3, seed=seed)
    rng = random.Random(seed)
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    epsilon = train_epsilon
    checkpoints: list[Checkpoint] = []
    td_sum = 0.0
    reward_sum = 0.0
    interval_td = 0.0
    interval_reward = 0.0
    interval_count = 0

    for step in range(1, environment.max_steps + 1):
        action = _choose_action(scores, epsilon, rng)
        activity = reservoir.action_activity(action)
        result = environment.step(action)
        reward_sum += result.reward
        interval_reward += result.reward
        next_scores = (0.0, 0.0, 0.0) if result.done else reservoir.step(result.observation)
        target = result.reward if result.done else result.reward + discount * max(next_scores)
        td_error = max(-1.0, min(1.0, target - scores[action]))
        reservoir.update_readout(action, td_error, learning_rate, activity)
        td_sum += abs(td_error)
        interval_td += abs(td_error)
        interval_count += 1
        scores = next_scores
        epsilon = max(min_epsilon, epsilon * train_epsilon_decay)

        if step % checkpoint_interval == 0 or result.done:
            checkpoints.append(Checkpoint(step, epsilon, scores, interval_td / max(1, interval_count),
                                          interval_reward / max(1, interval_count)))
            interval_td = 0.0
            interval_reward = 0.0
            interval_count = 0
        if result.done:
            break

    reservoir_stats = _measure_reservoir(reservoir, candles)
    readout = _measure_readout(reservoir, candles)
    trained_actions = _greedy_actions(reservoir, candles)

    random_circuit = build_random_topology(circuit, seed=seed + 1)
    random_reservoir = MaleCNSReservoir(random_circuit, feature_count=12, action_count=3, seed=seed)
    _train_readout(random_reservoir, candles, seed=seed, learning_rate=learning_rate,
                   discount=discount, epsilon=train_epsilon, epsilon_decay=train_epsilon_decay,
                   min_epsilon=min_epsilon)
    random_actions = _greedy_actions(random_reservoir, candles)

    return V11Diagnostics(tuple(checkpoints), reservoir_stats, readout, trained_actions,
                          random_actions, len(random_circuit.edges))


def _train_readout(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...], *, seed: int,
                   learning_rate: float, discount: float, epsilon: float,
                   epsilon_decay: float, min_epsilon: float) -> None:
    environment = CryptoTradingEnvironment(candles)
    rng = random.Random(seed)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    for _ in range(environment.max_steps):
        action = _choose_action(scores, epsilon, rng)
        activity = reservoir.action_activity(action)
        result = environment.step(action)
        next_scores = (0.0, 0.0, 0.0) if result.done else reservoir.step(result.observation)
        target = result.reward if result.done else result.reward + discount * max(next_scores)
        td_error = max(-1.0, min(1.0, target - scores[action]))
        reservoir.update_readout(action, td_error, learning_rate, activity)
        scores = next_scores
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if result.done:
            break


def _greedy_actions(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...]) -> tuple[int, int, int]:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    scores = reservoir.step(observation)
    counts = [0, 0, 0]
    for _ in range(environment.max_steps):
        action = max(range(3), key=scores.__getitem__)
        counts[action] += 1
        result = environment.step(action)
        if result.done:
            break
        scores = reservoir.step(result.observation)
    return tuple(counts)


def _measure_reservoir(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...]) -> ReservoirDiagnostics:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    activities: list[float] = []
    active_counts: list[int] = []
    states: set[tuple[float, ...]] = set()
    saturated = 0
    total_values = 0
    for _ in range(environment.max_steps):
        reservoir.step(observation)
        state = tuple(reservoir._state)
        states.add(tuple(round(value, 6) for value in state))
        activities.append(sum(abs(value) for value in state) / max(1, len(state)))
        active_counts.append(sum(abs(value) > 1e-3 for value in state))
        saturated += sum(abs(value) > 0.99 for value in state)
        total_values += len(state)
        result = environment.step(0)
        if result.done:
            break
        observation = result.observation
    mean = sum(activities) / max(1, len(activities))
    variance = sum((value - mean) ** 2 for value in activities) / max(1, len(activities))
    return ReservoirDiagnostics(mean, variance ** 0.5,
                               sum(active_counts) / max(1, len(active_counts)), len(states),
                               100.0 * saturated / max(1, total_values))


def _measure_readout(reservoir: MaleCNSReservoir, candles: tuple[Candle, ...]) -> ReadoutDiagnostics:
    environment = CryptoTradingEnvironment(candles)
    observation = environment.reset()
    reservoir.reset()
    margins = [[], [], []]
    for _ in range(environment.max_steps):
        scores = reservoir.step(observation)
        best = max(scores)
        for action in range(3):
            margins[action].append(scores[action] - best)
        result = environment.step(0)
        if result.done:
            break
        observation = result.observation
    margin_means = tuple(sum(values) / max(1, len(values)) for values in margins)
    return ReadoutDiagnostics(tuple(reservoir._readout), tuple(reservoir._readout_bias), margin_means)


def _choose_action(scores: tuple[float, ...], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(scores))
    return max(range(len(scores)), key=scores.__getitem__)
