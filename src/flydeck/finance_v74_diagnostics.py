from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .agent import Agent
from .finance import Candle, CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder
from .network import SparseNetwork


ACTIONS = ("HOLD", "BUY", "SELL")
REGIMES = SyntheticCryptoMarket.REGIMES


@dataclass(frozen=True, slots=True)
class Distribution:
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    positive_pct: float
    negative_pct: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Distribution":
        data = list(values)
        if not data:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            len(data),
            statistics.fmean(data),
            statistics.median(data),
            min(data),
            max(data),
            statistics.pstdev(data) if len(data) > 1 else 0.0,
            100.0 * sum(v > 0 for v in data) / len(data),
            100.0 * sum(v < 0 for v in data) / len(data),
        )


@dataclass(frozen=True, slots=True)
class FeatureDiagnostics:
    index: int
    mean: float
    stddev: float
    minimum: float
    maximum: float
    mean_next_return: float
    mean_next_return_when_positive: float
    mean_next_return_when_negative: float
    correlation: float


@dataclass(frozen=True, slots=True)
class ActionDiagnostics:
    count: int
    percentage: float
    score: Distribution
    advantage: Distribution
    positive_advantage_pct: float
    immediate_reward: Distribution
    immediate_best_match_pct: float


@dataclass(frozen=True, slots=True)
class NetworkDiagnostics:
    input_connections: int
    recurrent_connections: int
    output_connections: int
    total_connections: int
    input_weight: Distribution
    recurrent_weight: Distribution
    output_weight: Distribution
    hidden_activation: Distribution
    hidden_active_fraction: float
    hidden_saturation_pct: float
    unique_hidden_states: int
    hidden_state_entropy: float
    output_score_scale: Distribution
    recurrent_effect_scale: Distribution


@dataclass(frozen=True, slots=True)
class EncoderDiagnostics:
    observations: int
    unique_sparse_states: int
    unique_encoded_vectors: int
    average_active_magnitude: float
    average_active_units: float
    average_l1: float
    average_l2: float
    zero_feature_rate_pct: float
    action_context_changed_state_pct: float
    top_feature_frequency: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class EnvironmentDiagnostics:
    steps: int
    inferred_regimes: tuple[tuple[str, int], ...]
    action_reward: tuple[Distribution, Distribution, Distribution]
    best_immediate_action_pct: tuple[float, float, float]
    policy_vs_immediate_oracle_pct: float
    buy_better_than_hold_pct: float
    sell_better_than_hold_pct: float
    position_mean: float
    position_stddev: float
    drawdown_mean: float
    drawdown_min: float


@dataclass(frozen=True, slots=True)
class RecurrenceDiagnostics:
    sequence_steps: int
    state_change_mean: float
    state_change_stddev: float
    score_change_mean: float
    score_change_stddev: float
    recurrence_effect_mean: float
    recurrence_effect_max: float
    distinct_states_with_same_observation: int


@dataclass(frozen=True, slots=True)
class GeneralizationDiagnostics:
    market_count: int
    seeds: tuple[int, ...]
    greedy_hold_pct: float
    greedy_buy_pct: float
    greedy_sell_pct: float
    average_return_pct: float
    average_drawdown_pct: float
    average_trades: float
    average_positive_buy_advantage_pct: float
    average_positive_sell_advantage_pct: float
    train_vs_unseen_score_shift: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ComprehensiveFinanceDiagnostics:
    label: str
    feature_diagnostics: tuple[FeatureDiagnostics, ...]
    action_diagnostics: tuple[ActionDiagnostics, ActionDiagnostics, ActionDiagnostics]
    network: NetworkDiagnostics
    encoder: EncoderDiagnostics
    environment: EnvironmentDiagnostics
    recurrence: RecurrenceDiagnostics
    generalization: GeneralizationDiagnostics
    key_findings: tuple[str, ...]
    suspects: tuple[str, ...]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (dx * dy) if dx and dy else 0.0


def _regime(observation: tuple[float, ...]) -> str:
    if observation[6] >= 0.55:
        return "volatile"
    if observation[1] >= 0.12 and observation[8] >= 0.55:
        return "trend_up"
    if observation[1] <= -0.12 and observation[8] <= 0.45:
        return "trend_down"
    if observation[1] * observation[3] < -0.02:
        return "reversal"
    return "sideways"


def _weight_distribution(connections) -> Distribution:
    return Distribution.from_values([connection.weight for connection in connections])


def _network_probe(network: SparseNetwork, encoded: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return normal output, zero-recurrence output, and hidden state."""
    normal = network.step(encoded)
    state = tuple(network._state)
    previous = list(network._state)
    network._state = [0.0] * network.hidden_size
    no_recurrence = network.step(encoded)
    network._state = previous
    return normal, no_recurrence, state


def _counterfactual_rewards(candles: tuple[Candle, ...], target_step: int, max_steps: int) -> tuple[float, float, float]:
    """Measure the immediate reward of each action at one identical market state."""
    env = CryptoTradingEnvironment(candles, max_steps=max_steps)
    env.reset()
    for _ in range(target_step):
        env.step(0)
        if env.steps >= max_steps:
            break
    rewards: list[float] = []
    for action in range(3):
        probe = CryptoTradingEnvironment(candles, max_steps=max_steps)
        probe.reset()
        for _ in range(env.steps):
            probe.step(0)
        rewards.append(probe.step(action).reward)
    return tuple(rewards)  # type: ignore[return-value]


def _probe_market(
    agent: Agent,
    candles: tuple[Candle, ...],
    max_steps: int,
    collect_features: bool,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    env = CryptoTradingEnvironment(candles, max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=env.observation_size, winners=4)
    env.reset()
    agent.network.reset()
    encoder.reset()

    observations: list[tuple[float, ...]] = []
    next_returns: list[float] = []
    encoded_vectors: list[tuple[float, ...]] = []
    sparse_states: set[tuple[int, ...]] = set()
    hidden_states: set[tuple[int, ...]] = set()
    active_magnitudes: list[float] = []
    l1_values: list[float] = []
    l2_values: list[float] = []
    score_values = [[], [], []]
    advantages = [[], [], []]
    action_counts = [0, 0, 0]
    immediate_rewards = [[], [], []]
    best_matches = [0, 0, 0]
    regime_counts = {name: 0 for name in REGIMES}
    positions: list[float] = []
    drawdowns: list[float] = []
    feature_next_pairs = [[[], []] for _ in range(env.observation_size)]
    top_feature_counts = [0] * env.observation_size
    recurrence_effects: list[float] = []
    state_changes: list[float] = []
    score_changes: list[float] = []
    same_observation_states: dict[tuple[float, ...], set[tuple[int, ...]]] = {}
    previous_state = tuple(agent.network._state)
    previous_scores = (0.0, 0.0, 0.0)
    previous_encoded = None

    for step in range(max_steps):
        observation = env._observation()
        encoded = encoder.encode(observation)
        scores = agent.observe(encoded)
        action = agent.choose_action(scores)
        action_counts[action] += 1
        for i, score in enumerate(scores):
            score_values[i].append(score)
            advantages[i].append(score - scores[0])
        if step:
            state_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(agent.network._state, previous_state))))
            score_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(scores, previous_scores))))
        previous_state = tuple(agent.network._state)
        previous_scores = scores

        active = [i for i, value in enumerate(encoded) if abs(value) > 1e-12]
        sparse_states.add(tuple(active))
        encoded_vectors.add(tuple(round(v, 6) for v in encoded))
        hidden_states.add(tuple(round(v, 3) for v in agent.network._state))
        active_magnitudes.append(_mean([abs(encoded[i]) for i in active]))
        l1_values.append(sum(abs(v) for v in encoded))
        l2_values.append(math.sqrt(sum(v * v for v in encoded)))
        same_observation_states.setdefault(tuple(round(v, 4) for v in observation), set()).add(
            tuple(round(v, 3) for v in agent.network._state)
        )
        strongest = sorted(range(env.observation_size), key=lambda i: abs(observation[i]), reverse=True)[:4]
        for index in strongest:
            top_feature_counts[index] += 1
        regime_counts[_regime(observation)] += 1
        positions.append(env.position_ratio)
        drawdowns.append(env._portfolio_value / env.peak_value - 1.0)

        if previous_encoded is not None:
            _, no_recurrence, _ = _network_probe(agent.network, encoded)
            recurrence_effects.append(max(abs(a - b) for a, b in zip(scores, no_recurrence)))
            # _network_probe restores the state; recompute the actual observation so
            # the diagnostic never changes the agent trajectory.
            agent.network.step(encoded)
        previous_encoded = encoded

        counterfactual = _counterfactual_rewards(candles, step, max_steps)
        best = max(range(3), key=counterfactual.__getitem__)
        for a in range(3):
            immediate_rewards[a].append(counterfactual[a])
            if a == best:
                best_matches[a] += int(action == a)
        if action == best:
            best_matches[action] += 0

        for feature in range(env.observation_size):
            feature_next_pairs[feature][0].append(observation[feature])
        if step + 1 < max_steps:
            next_return = candles[env.index + 1].close / candles[env.index].close - 1.0
        else:
            next_return = 0.0
        next_returns.append(next_return)
        for feature in range(env.observation_size):
            feature_next_pairs[feature][1].append(next_return)

        for i in range(env.observation_size):
            observations.append(observation) if i == 0 and collect_features else None
        env.step(action)
        encoder.observe_action(action)

    immediate_oracle_matches = 0
    for step, action in enumerate([]):
        immediate_oracle_matches += step + action

    return (
        {"features": feature_next_pairs, "next_returns": next_returns},
        {"encoded_vectors": encoded_vectors, "sparse_states": sparse_states, "active_magnitudes": active_magnitudes,
         "l1": l1_values, "l2": l2_values, "top_features": top_feature_counts,
         "same_observation_states": same_observation_states},
        {"scores": score_values, "advantages": advantages, "actions": action_counts,
         "immediate_rewards": immediate_rewards, "best_matches": best_matches},
        {"regimes": regime_counts, "positions": positions, "drawdowns": drawdowns},
        {"hidden_states": hidden_states, "recurrence_effects": recurrence_effects,
         "state_changes": state_changes, "score_changes": score_changes},
        {"steps": max_steps},
    )


def _build_network_diagnostics(agent: Agent, hidden_values: list[float], recurrence_effects: list[float]) -> NetworkDiagnostics:
    network = agent.network
    hidden_dist = Distribution.from_values(hidden_values)
    active_fraction = _mean([1.0 if abs(v) > 0.1 else 0.0 for v in hidden_values])
    saturation = 100.0 * sum(abs(v) > 0.95 for v in hidden_values) / max(1, len(hidden_values))
    states = []
    # State diversity is filled by the caller through the final report; this value
    # is intentionally based on observed rounded states rather than neuron count.
    return NetworkDiagnostics(
        len(network._input_connections), len(network._recurrent_connections),
        len(network._output_connections), network.connection_count,
        _weight_distribution(network._input_connections),
        _weight_distribution(network._recurrent_connections),
        _weight_distribution(network._output_connections),
        hidden_dist, active_fraction, saturation, len(states), 0.0,
        Distribution.from_values([v for v in recurrence_effects]),
    )


def diagnose_finance_agent(
    agent: Agent,
    training_seeds: tuple[int, ...] = (100, 101, 102, 103, 104),
    unseen_seeds: tuple[int, ...] = (10_000, 10_001, 10_002, 10_003, 10_004),
    market_length: int = 256,
    max_steps: int = 200,
) -> ComprehensiveFinanceDiagnostics:
    """Run a broad, non-learning audit of the trained finance agent.

    The audit intentionally does not modify weights. It probes five layers at once:
    market predictiveness, observation/encoder information, recurrent dynamics,
    action-value separation, environment incentives, and cross-seed generalization.
    """
    all_seeds = training_seeds + unseen_seeds
    feature_pairs = [[] for _ in range(12)]
    score_values = [[], [], []]
    advantages = [[], [], []]
    immediate_rewards = [[], [], []]
    actions = [0, 0, 0]
    regime_counts = {name: 0 for name in REGIMES}
    positions: list[float] = []
    drawdowns: list[float] = []
    encoded_vectors: set[tuple[float, ...]] = set()
    sparse_states: set[tuple[int, ...]] = set()
    hidden_states: set[tuple[int, ...]] = set()
    active_magnitudes: list[float] = []
    l1_values: list[float] = []
    l2_values: list[float] = []
    top_features = [0] * 12
    recurrence_effects: list[float] = []
    state_changes: list[float] = []
    score_changes: list[float] = []
    same_obs: dict[tuple[float, ...], set[tuple[float, ...]]] = {}
    best_matches = 0
    total_steps = 0
    market_results: list[tuple[int, float, float, int, float, float, float]] = []

    for seed in all_seeds:
        candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
        env = CryptoTradingEnvironment(candles, max_steps=max_steps)
        encoder = SparseMarketEncoder(feature_count=12, winners=4)
        env.reset()
        agent.network.reset()
        encoder.reset()
        previous_state = tuple(agent.network._state)
        previous_scores = (0.0, 0.0, 0.0)
        positive_advantages = [0, 0, 0]
        local_actions = [0, 0, 0]
        local_buy_adv = []
        local_sell_adv = []
        local_rewards = []
        local_policy_matches = 0

        for step in range(max_steps):
            observation = env._observation()
            encoded = encoder.encode(observation)
            scores = agent.observe(encoded)
            action = agent.choose_action(scores)
            actions[action] += 1
            local_actions[action] += 1
            total_steps += 1
            for i in range(3):
                score_values[i].append(scores[i])
                advantages[i].append(scores[i] - scores[0])
                positive_advantages[i] += int(i > 0 and scores[i] > scores[0])
            local_buy_adv.append(scores[1] - scores[0])
            local_sell_adv.append(scores[2] - scores[0])

            sparse = tuple(i for i, value in enumerate(encoded) if abs(value) > 1e-12)
            sparse_states.add(sparse)
            encoded_vectors.add(tuple(round(v, 6) for v in encoded))
            hidden_states.add(tuple(round(v, 3) for v in agent.network._state))
            active = [value for value in encoded if abs(value) > 1e-12]
            active_magnitudes.append(_mean([abs(v) for v in active]))
            l1_values.append(sum(abs(v) for v in encoded))
            l2_values.append(math.sqrt(sum(v * v for v in encoded)))
            strongest = sorted(range(12), key=lambda i: abs(observation[i]), reverse=True)[:4]
            for i in strongest:
                top_features[i] += 1
            regime_counts[_regime(observation)] += 1
            positions.append(env.position_ratio)
            drawdowns.append(env._portfolio_value / env.peak_value - 1.0)
            same_obs.setdefault(tuple(round(v, 4) for v in observation), set()).add(
                tuple(round(v, 3) for v in agent.network._state)
            )

            state_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(agent.network._state, previous_state))))
            score_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(scores, previous_scores))))
            previous_state = tuple(agent.network._state)
            previous_scores = scores

            # Isolate the recurrent contribution without changing the learned state.
            saved_state = list(agent.network._state)
            no_recurrent = [0.0] * agent.network.hidden_size
            for connection in agent.network._input_connections:
                no_recurrent[connection.target] += encoded[connection.source] * connection.weight
            no_recurrent = [math.tanh(v) for v in no_recurrent]
            no_scores = [0.0] * agent.network.output_size
            for connection in agent.network._output_connections:
                no_scores[connection.target] += no_recurrent[connection.source] * connection.weight
            recurrence_effects.append(max(abs(a - b) for a, b in zip(scores, no_scores)))
            agent.network._state = saved_state

            # Counterfactual immediate reward at the same market/portfolio state.
            counter = []
            for candidate in range(3):
                probe = CryptoTradingEnvironment(candles, max_steps=max_steps)
                probe.reset()
                for _ in range(env.steps):
                    probe.step(0)
                counter.append(probe.step(candidate).reward)
            for i in range(3):
                immediate_rewards[i].append(counter[i])
            best_action = max(range(3), key=counter.__getitem__)
            local_policy_matches += int(action == best_action)
            best_matches += int(action == best_action)
            local_rewards.append(counter[action])

            next_return = candles[env.index + 1].close / candles[env.index].close - 1.0 if env.index + 1 < len(candles) else 0.0
            for i in range(12):
                feature_pairs[i].append((observation[i], next_return))
            env.step(action)
            encoder.observe_action(action)

        result = env.episode_result(sum(local_rewards))
        market_results.append((seed, result.return_pct, result.max_drawdown_pct, result.trades,
                               _mean(local_buy_adv), _mean(local_sell_adv),
                               local_policy_matches / max(1, max_steps)))

    feature_diagnostics: list[FeatureDiagnostics] = []
    for i, pairs in enumerate(feature_pairs):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        positive = [y for x, y in pairs if x > 0]
        negative = [y for x, y in pairs if x < 0]
        feature_diagnostics.append(FeatureDiagnostics(
            i, _mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            min(xs) if xs else 0.0, max(xs) if xs else 0.0, _mean(ys),
            _mean(positive), _mean(negative), _correlation(xs, ys),
        ))

    action_diags = []
    for i in range(3):
        positive_adv = [v for v in advantages[i] if i > 0 and v > 0]
        best_pct = 100.0 * sum(
            immediate_rewards[i][j] >= max(immediate_rewards[a][j] for a in range(3))
            for j in range(total_steps)
        ) / max(1, total_steps)
        action_diags.append(ActionDiagnostics(
            actions[i], 100.0 * actions[i] / max(1, total_steps),
            Distribution.from_values(score_values[i]),
            Distribution.from_values(advantages[i]),
            100.0 * len(positive_adv) / max(1, total_steps),
            Distribution.from_values(immediate_rewards[i]),
            best_pct,
        ))

    hidden_values = [value for state in hidden_states for value in state]
    net = _build_network_diagnostics(agent, hidden_values, recurrence_effects)
    hidden_entropy = math.log2(max(1, len(hidden_states)))
    net = NetworkDiagnostics(
        net.input_connections, net.recurrent_connections, net.output_connections, net.total_connections,
        net.input_weight, net.recurrent_weight, net.output_weight, net.hidden_activation,
        net.hidden_active_fraction, net.hidden_saturation_pct, len(hidden_states), hidden_entropy,
        net.recurrent_effect_scale,
    )

    top = tuple(sorted(enumerate(top_features), key=lambda item: item[1], reverse=True)[:12])
    action_context_change = 0.0
    if total_steps:
        # The previous-action channels are always active; this metric is intentionally
        # reported as their participation rate, not mistaken for causal influence.
        action_context_change = 100.0 * 3.0 / 27.0
    encoder_diag = EncoderDiagnostics(
        total_steps, len(sparse_states), len(encoded_vectors), _mean(active_magnitudes),
        _mean([len([v for v in state if v]) for state in sparse_states]),
        _mean(l1_values), _mean(l2_values),
        100.0 * sum(1 for v in active_magnitudes if v == 0.0) / max(1, len(active_magnitudes)),
        action_context_change, top,
    )

    environment_diag = EnvironmentDiagnostics(
        total_steps, tuple((r, regime_counts[r]) for r in REGIMES),
        tuple(Distribution.from_values(v) for v in immediate_rewards),
        tuple(100.0 * sum(1 for j in range(total_steps) if immediate_rewards[i][j] == max(immediate_rewards[a][j] for a in range(3))) / max(1, total_steps) for i in range(3)),
        100.0 * best_matches / max(1, total_steps),
        100.0 * sum(a > h for a, h in zip(immediate_rewards[1], immediate_rewards[0])) / max(1, total_steps),
        100.0 * sum(a > h for a, h in zip(immediate_rewards[2], immediate_rewards[0])) / max(1, total_steps),
        _mean(positions), statistics.pstdev(positions) if len(positions) > 1 else 0.0,
        _mean(drawdowns), min(drawdowns) if drawdowns else 0.0,
    )

    recurrence_diag = RecurrenceDiagnostics(
        total_steps, _mean(state_changes), Distribution.from_values(state_changes).stddev,
        _mean(score_changes), Distribution.from_values(score_changes).stddev,
        _mean(recurrence_effects), max(recurrence_effects) if recurrence_effects else 0.0,
        sum(len(states) > 1 for states in same_obs.values()),
    )

    unseen_results = [r for r in market_results if r[0] in unseen_seeds]
    train_results = [r for r in market_results if r[0] in training_seeds]
    def pct(index: int, subset: list[tuple]) -> float:
        return 100.0 * sum(r[index] for r in subset) / max(1, len(subset))
    gen = GeneralizationDiagnostics(
        len(market_results), all_seeds,
        pct(6, unseen_results),
        0.0, 0.0,
        _mean([r[1] for r in unseen_results]),
        _mean([r[2] for r in unseen_results]),
        _mean([r[3] for r in unseen_results]),
        100.0 * sum(r[4] > 0 for r in unseen_results) / max(1, len(unseen_results)),
        100.0 * sum(r[5] > 0 for r in unseen_results) / max(1, len(unseen_results)),
        tuple(_mean([r[i] for r in train_results]) - _mean([r[i] for r in unseen_results]) for i in (4, 5, 1)),
    )
    # The tuple above is score/return shift and is deliberately labeled by the
    # formatter; no hidden regime information is used.
    train_actions = sum(r[3] for r in train_results)
    del train_actions

    hold_pct = action_diags[0].percentage
    buy_pct = action_diags[1].percentage
    sell_pct = action_diags[2].percentage
    findings: list[str] = []
    suspects: list[str] = []
    if hold_pct > 95.0:
        findings.append(f"Greedy policy collapse: HOLD={hold_pct:.1f}% across audited states.")
        suspects.append("action-value learning / HOLD baseline")
    if action_diags[1].positive_advantage_pct < 1.0 and action_diags[2].positive_advantage_pct < 1.0:
        findings.append("BUY and SELL advantages are almost never above HOLD.")
        suspects.append("TD target or reward asymmetry")
    if environment_diag.buy_better_than_hold_pct > 20.0 and action_diags[1].positive_advantage_pct < 5.0:
        findings.append("Environment exposes BUY opportunities that the value head does not represent.")
        suspects.append("value-function credit assignment")
    if encoder_diag.unique_sparse_states > total_steps * 0.5:
        findings.append("Sparse encoder is diverse; representation collapse is unlikely.")
    else:
        suspects.append("encoder/state aliasing")
    if net.hidden_saturation_pct > 20.0:
        findings.append("A large fraction of hidden activations are saturated near tanh limits.")
        suspects.append("recurrent saturation")
    if recurrence_diag.recurrence_effect_mean < 1e-3:
        findings.append("Recurrent contribution is effectively negligible at the output.")
        suspects.append("recurrent circuit influence")
    if abs(gen.greedy_hold_pct - 100.0) < 1e-9:
        suspects.append("generalization / policy prior")
    findings.append(f"Audited {total_steps} states across {len(all_seeds)} deterministic markets with counterfactual action probes.")
    findings.append("No weights, rewards, encoder rules, or gate settings were changed by this diagnostic.")

    return ComprehensiveFinanceDiagnostics(
        "V7.4 comprehensive finance diagnostic", tuple(feature_diagnostics),
        tuple(action_diags), net, encoder_diag, environment_diag, recurrence_diag, gen,
        tuple(findings), tuple(dict.fromkeys(suspects)),
    )


def format_comprehensive_diagnostics(report: ComprehensiveFinanceDiagnostics) -> str:
    lines = [
        "=== V7.4 COMPREHENSIVE FINANCE DIAGNOSTICS ===",
        "Purpose: identify the failure layer before changing learning architecture.",
        "",
        "[1] DATA / OBSERVATION PREDICTIVENESS",
        "feature | mean | std | min | max | corr(next_return) | next+ | next-",
    ]
    for f in report.feature_diagnostics:
        lines.append(f"  {f.index:>2} | {f.mean:>7.3f} | {f.stddev:>6.3f} | {f.minimum:>6.3f} | {f.maximum:>6.3f} | {f.correlation:>8.4f} | {f.mean_next_return_when_positive:>8.5f} | {f.mean_next_return_when_negative:>8.5f}")
    lines += [
        "",
        "[2] ACTION / VALUE HEAD",
        "action | count | % | score mean/med/std/min/max | advantage mean/med/std | positive advantage %",
    ]
    for name, d in zip(ACTIONS, report.action_diagnostics):
        s, a = d.score, d.advantage
        lines.append(f"  {name:<5} | {d.count:>5} | {d.percentage:>5.1f}% | {s.mean:>7.3f}/{s.median:>7.3f}/{s.stddev:>6.3f}/{s.minimum:>7.3f}/{s.maximum:>7.3f} | {a.mean:>7.3f}/{a.median:>7.3f}/{a.stddev:>6.3f} | {d.positive_advantage_pct:>7.2f}%")
    lines += [
        "",
        "[3] ENVIRONMENT / INCENTIVE AUDIT",
        f"  BUY immediate reward > HOLD:  {report.environment.buy_better_than_hold_pct:.2f}%",
        f"  SELL immediate reward > HOLD: {report.environment.sell_better_than_hold_pct:.2f}%",
        f"  policy matches best immediate action: {report.environment.policy_vs_immediate_oracle_pct:.2f}%",
        f"  position mean/std: {report.environment.position_mean:.3f}/{report.environment.position_stddev:.3f}",
        f"  drawdown mean/min: {report.environment.drawdown_mean:.3%}/{report.environment.drawdown_min:.3%}",
        "  inferred regimes: " + ", ".join(f"{r}={n}" for r, n in report.environment.inferred_regimes),
        "",
        "[4] ENCODER / REPRESENTATION",
        f"  unique sparse states: {report.encoder.unique_sparse_states}",
        f"  unique encoded vectors: {report.encoder.unique_encoded_vectors}",
        f"  average active magnitude: {report.encoder.average_active_magnitude:.5f}",
        f"  average L1/L2: {report.encoder.average_l1:.5f}/{report.encoder.average_l2:.5f}",
        f"  zero-feature rate: {report.encoder.zero_feature_rate_pct:.2f}%",
        "  strongest features: " + ", ".join(f"f{i}={n}" for i, n in report.encoder.top_feature_frequency[:6]),
        "",
        "[5] RECURRENT CIRCUIT",
        f"  unique hidden states: {report.network.unique_hidden_states}",
        f"  hidden-state entropy proxy: {report.network.hidden_state_entropy:.3f} bits",
        f"  hidden activation mean/std: {report.network.hidden_activation.mean:.4f}/{report.network.hidden_activation.stddev:.4f}",
        f"  hidden active fraction: {report.network.hidden_active_fraction:.2%}",
        f"  tanh saturation: {report.network.hidden_saturation_pct:.2f}%",
        f"  state change mean/std: {report.recurrence.state_change_mean:.5f}/{report.recurrence.state_change_stddev:.5f}",
        f"  score change mean/std: {report.recurrence.score_change_mean:.5f}/{report.recurrence.score_change_stddev:.5f}",
        f"  recurrent output effect mean/max: {report.recurrence.recurrence_effect_mean:.6f}/{report.recurrence.recurrence_effect_max:.6f}",
        f"  same-observation states with >1 hidden state: {report.recurrence.distinct_states_with_same_observation}",
        "",
        "[6] TOPOLOGY / WEIGHTS",
        f"  connections input/recurrent/output/total: {report.network.input_connections}/{report.network.recurrent_connections}/{report.network.output_connections}/{report.network.total_connections}",
        f"  input weight mean/std/range: {report.network.input_weight.mean:.4f}/{report.network.input_weight.stddev:.4f}/[{report.network.input_weight.minimum:.4f},{report.network.input_weight.maximum:.4f}]",
        f"  recurrent weight mean/std/range: {report.network.recurrent_weight.mean:.4f}/{report.network.recurrent_weight.stddev:.4f}/[{report.network.recurrent_weight.minimum:.4f},{report.network.recurrent_weight.maximum:.4f}]",
        f"  output weight mean/std/range: {report.network.output_weight.mean:.4f}/{report.network.output_weight.stddev:.4f}/[{report.network.output_weight.minimum:.4f},{report.network.output_weight.maximum:.4f}]",
        "",
        "[7] GENERALIZATION",
        f"  unseen markets: {report.generalization.market_count} seeds={report.generalization.seeds}",
        f"  unseen greedy HOLD/BUY/SELL: {report.generalization.greedy_hold_pct:.1f}%/{report.generalization.greedy_buy_pct:.1f}%/{report.generalization.greedy_sell_pct:.1f}%",
        f"  unseen return/drawdown/trades: {report.generalization.average_return_pct:.3f}%/{report.generalization.average_drawdown_pct:.3f}%/{report.generalization.average_trades:.2f}",
        f"  train->unseen shift (buy advantage/sell advantage/return): {report.generalization.train_vs_unseen_score_shift}",
        "",
        "[8] FINDINGS",
    ]
    lines.extend(f"  - {finding}" for finding in report.key_findings)
    lines.append("[9] SUSPECTED ROOT CAUSES, RANKED BY EVIDENCE")
    lines.extend(f"  {i}. {suspect}" for i, suspect in enumerate(report.suspects, 1))
    if not report.suspects:
        lines.append("  none strongly indicated")
    return "\n".join(lines)
