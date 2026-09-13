from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .agent import Agent
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder

ACTIONS = ("HOLD", "BUY", "SELL")
REGIMES = SyntheticCryptoMarket.REGIMES


@dataclass(frozen=True, slots=True)
class Distribution:
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    positive_pct: float
    negative_pct: float

    @classmethod
    def make(cls, values: list[float]) -> "Distribution":
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            statistics.fmean(values), statistics.median(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0,
            min(values), max(values),
            100.0 * sum(v > 0 for v in values) / len(values),
            100.0 * sum(v < 0 for v in values) / len(values),
        )


@dataclass(frozen=True, slots=True)
class FinanceDiagnosticReport:
    states: int
    markets: int
    action_counts: tuple[int, int, int]
    score_distributions: tuple[Distribution, Distribution, Distribution]
    advantage_distributions: tuple[Distribution, Distribution, Distribution]
    positive_advantage_pct: tuple[float, float, float]
    immediate_reward_distributions: tuple[Distribution, Distribution, Distribution]
    immediate_best_action_pct: tuple[float, float, float]
    policy_immediate_match_pct: float
    buy_better_than_hold_pct: float
    sell_better_than_hold_pct: float
    feature_stats: tuple[Distribution, ...]
    feature_next_return_correlation: tuple[float, ...]
    sparse_unique_states: int
    encoded_unique_vectors: int
    active_magnitude: Distribution
    hidden_unique_states: int
    hidden_activation: Distribution
    hidden_saturation_pct: float
    recurrent_effect: Distribution
    state_change: Distribution
    score_change: Distribution
    same_observation_multiple_hidden_states: int
    input_weight: Distribution
    recurrent_weight: Distribution
    output_weight: Distribution
    inferred_regimes: tuple[tuple[str, int], ...]
    position_distribution: Distribution
    drawdown_distribution: Distribution
    unseen_action_pct: tuple[float, float, float]
    unseen_return_pct: float
    unseen_drawdown_pct: float
    unseen_trades: float
    findings: tuple[str, ...]
    suspects: tuple[str, ...]


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (dx * dy) if dx and dy else 0.0


def _regime(obs: tuple[float, ...]) -> str:
    if obs[6] >= 0.55:
        return "volatile"
    if obs[1] >= 0.12 and obs[8] >= 0.55:
        return "trend_up"
    if obs[1] <= -0.12 and obs[8] <= 0.45:
        return "trend_down"
    if obs[1] * obs[3] < -0.02:
        return "reversal"
    return "sideways"


def _weights(connections) -> Distribution:
    return Distribution.make([c.weight for c in connections])


def _step_without_recurrence(agent: Agent, encoded: tuple[float, ...]) -> tuple[float, ...]:
    hidden = [0.0] * agent.network.hidden_size
    for c in agent.network._input_connections:
        hidden[c.target] += encoded[c.source] * c.weight
    hidden = [math.tanh(v) for v in hidden]
    output = [0.0] * agent.network.output_size
    for c in agent.network._output_connections:
        output[c.target] += hidden[c.source] * c.weight
    return tuple(output)


def _counterfactual(candles, step: int, max_steps: int) -> tuple[float, float, float]:
    """Immediate rewards for all actions at exactly the same portfolio/market state."""
    result = []
    for action in range(3):
        env = CryptoTradingEnvironment(candles, max_steps=max_steps)
        env.reset()
        for _ in range(step):
            env.step(0)
        result.append(env.step(action).reward)
    return tuple(result)


def diagnose_finance_agent(
    agent: Agent,
    training_seeds: tuple[int, ...] = (100, 101, 102, 103, 104),
    unseen_seeds: tuple[int, ...] = (10_000, 10_001, 10_002, 10_003, 10_004),
    market_length: int = 256,
    max_steps: int = 200,
) -> FinanceDiagnosticReport:
    """Audit the entire finance stack without learning or changing any weights.

    It checks data signal, observation scaling, sparse encoding, recurrent dynamics,
    topology/weights, Q/action separation, immediate counterfactual rewards,
    portfolio incentives and cross-seed generalization. The report is deliberately
    diagnostic: it does not introduce a new training mechanism.
    """
    all_seeds = training_seeds + unseen_seeds
    feature_values = [[] for _ in range(12)]
    feature_returns = [[] for _ in range(12)]
    scores = [[], [], []]
    advantages = [[], [], []]
    immediate = [[], [], []]
    actions = [0, 0, 0]
    regimes = {r: 0 for r in REGIMES}
    sparse_states: set[tuple[int, ...]] = set()
    encoded_vectors: set[tuple[float, ...]] = set()
    hidden_states: set[tuple[float, ...]] = set()
    active_magnitudes: list[float] = []
    hidden_values: list[float] = []
    recurrence_effects: list[float] = []
    state_changes: list[float] = []
    score_changes: list[float] = []
    positions: list[float] = []
    drawdowns: list[float] = []
    same_observation: dict[tuple[float, ...], set[tuple[float, ...]]] = {}
    oracle_matches = 0
    total = 0
    unseen_actions = [0, 0, 0]
    unseen_returns: list[float] = []
    unseen_drawdowns: list[float] = []
    unseen_trades: list[int] = []

    for seed in all_seeds:
        candles = SyntheticCryptoMarket(length=market_length, seed=seed).generate()
        env = CryptoTradingEnvironment(candles, max_steps=max_steps)
        encoder = SparseMarketEncoder(feature_count=12, winners=4)
        env.reset(); agent.network.reset(); encoder.reset()
        previous_hidden = tuple(agent.network._state)
        previous_scores = (0.0, 0.0, 0.0)
        unseen = seed in unseen_seeds

        for step in range(max_steps):
            obs = env._observation()
            encoded = encoder.encode(obs)
            action_scores = agent.observe(encoded)
            action = agent.choose_action(action_scores)
            actions[action] += 1
            if unseen:
                unseen_actions[action] += 1
            total += 1

            for i in range(3):
                scores[i].append(action_scores[i])
                advantages[i].append(action_scores[i] - action_scores[0])
            sparse = tuple(i for i, v in enumerate(encoded) if abs(v) > 1e-12)
            sparse_states.add(sparse)
            encoded_vectors.add(tuple(round(v, 6) for v in encoded))
            hidden_states.add(tuple(round(v, 3) for v in agent.network._state))
            hidden_values.extend(agent.network._state)
            active = [abs(v) for v in encoded if abs(v) > 1e-12]
            active_magnitudes.append(statistics.fmean(active) if active else 0.0)
            same_observation.setdefault(tuple(round(v, 4) for v in obs), set()).add(tuple(round(v, 3) for v in agent.network._state))

            for i, value in enumerate(obs):
                feature_values[i].append(value)
                next_return = candles[env.index + 1].close / candles[env.index].close - 1.0 if env.index + 1 < len(candles) else 0.0
                feature_returns[i].append(next_return)
            regimes[_regime(obs)] += 1
            positions.append(env.position_ratio)
            drawdowns.append(env._portfolio_value / env.peak_value - 1.0)

            if step:
                state_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(agent.network._state, previous_hidden))))
                score_changes.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(action_scores, previous_scores))))
            previous_hidden = tuple(agent.network._state)
            previous_scores = action_scores

            no_recurrence = _step_without_recurrence(agent, encoded)
            recurrence_effects.append(max(abs(a - b) for a, b in zip(action_scores, no_recurrence)))

            counter = _counterfactual(candles, step, max_steps)
            best = max(range(3), key=counter.__getitem__)
            oracle_matches += int(action == best)
            for i in range(3):
                immediate[i].append(counter[i])
            env.step(action)
            encoder.observe_action(action)

        result = env.episode_result(0.0)
        if unseen:
            unseen_returns.append(result.return_pct)
            unseen_drawdowns.append(result.max_drawdown_pct)
            unseen_trades.append(result.trades)

    score_dist = tuple(Distribution.make(v) for v in scores)
    adv_dist = tuple(Distribution.make(v) for v in advantages)
    immediate_dist = tuple(Distribution.make(v) for v in immediate)
    positive_adv = tuple(100.0 * sum(v > 0 for v in advantages[i]) / max(1, total) for i in range(3))
    best_pct = tuple(100.0 * sum(immediate[i][j] == max(immediate[a][j] for a in range(3)) for j in range(total)) / max(1, total) for i in range(3))
    feature_stats = tuple(Distribution.make(v) for v in feature_values)
    feature_corr = tuple(_corr(feature_values[i], feature_returns[i]) for i in range(12))
    hidden_dist = Distribution.make(hidden_values)
    hidden_sat = 100.0 * sum(abs(v) > 0.95 for v in hidden_values) / max(1, len(hidden_values))
    inferred = tuple((r, regimes[r]) for r in REGIMES)
    unseen_total = max(1, sum(unseen_actions))
    unseen_action_pct = tuple(100.0 * n / unseen_total for n in unseen_actions)

    findings: list[str] = []
    suspects: list[str] = []
    if score_dist[0].mean > score_dist[1].mean and score_dist[0].mean > score_dist[2].mean:
        findings.append("HOLD has the highest mean learned value across the audited state distribution.")
        suspects.append("value head / TD target")
    if positive_adv[1] < 5.0 and positive_adv[2] < 5.0:
        findings.append("BUY and SELL almost never beat HOLD in learned advantage, indicating value suppression rather than a gate problem.")
        suspects.append("action-value credit assignment")
    buy_vs_hold = 100.0 * sum(a > h for a, h in zip(immediate[1], immediate[0])) / max(1, total)
    sell_vs_hold = 100.0 * sum(a > h for a, h in zip(immediate[2], immediate[0])) / max(1, total)
    if buy_vs_hold > positive_adv[1] + 10.0:
        findings.append("The environment contains materially more BUY-immediate opportunities than the value head recognizes.")
        suspects.append("reward/portfolio-state representation")
    if len(sparse_states) > total * 0.5:
        findings.append("Sparse representation is diverse; encoder collapse is unlikely.")
    else:
        suspects.append("encoder aliasing / information loss")
    if hidden_sat > 20.0:
        findings.append("Tanh hidden units show substantial saturation.")
        suspects.append("recurrent saturation")
    if Distribution.make(recurrence_effects).mean < 1e-3:
        findings.append("Recurrent connections have negligible output influence in the audited states.")
        suspects.append("recurrent circuit influence")
    if unseen_action_pct[0] > 95.0:
        findings.append("The greedy HOLD collapse persists across multiple unseen markets.")
        suspects.append("generalization / policy prior")
    if abs(score_dist[0].mean - score_dist[1].mean) > 1.0 and positive_adv[1] < 5.0:
        suspects.append("output initialization / persistent HOLD bias")
    findings.append(f"Counterfactual rewards were evaluated for all three actions at {total} identical state points.")
    findings.append("This diagnostic is non-destructive: it performs no learning and changes no weights or environment parameters.")

    return FinanceDiagnosticReport(
        total, len(all_seeds), tuple(actions), score_dist, adv_dist, positive_adv,
        immediate_dist, best_pct, 100.0 * oracle_matches / max(1, total),
        buy_vs_hold, sell_vs_hold, feature_stats, feature_corr,
        len(sparse_states), len(encoded_vectors), Distribution.make(active_magnitudes),
        len(hidden_states), hidden_dist, hidden_sat, Distribution.make(recurrence_effects),
        Distribution.make(state_changes), Distribution.make(score_changes),
        sum(len(states) > 1 for states in same_observation.values()),
        _weights(agent.network._input_connections), _weights(agent.network._recurrent_connections),
        _weights(agent.network._output_connections), inferred, Distribution.make(positions),
        Distribution.make(drawdowns), unseen_action_pct,
        statistics.fmean(unseen_returns) if unseen_returns else 0.0,
        statistics.fmean(unseen_drawdowns) if unseen_drawdowns else 0.0,
        statistics.fmean(unseen_trades) if unseen_trades else 0.0,
        tuple(findings), tuple(dict.fromkeys(suspects)),
    )


def format_diagnostic_report(report: FinanceDiagnosticReport) -> str:
    lines = ["=== V7.4 COMPREHENSIVE ROOT-CAUSE AUDIT ===", f"states: {report.states} | markets: {report.markets}", ""]
    lines += ["[1] LEARNED ACTION VALUES", "action | count | % | score mean/med/std/range | advantage mean/med/std | positive advantage"]
    for i, name in enumerate(ACTIONS):
        s, a = report.score_distributions[i], report.advantage_distributions[i]
        lines.append(f"  {name:<5} | {report.action_counts[i]:>5} | {100*report.action_counts[i]/report.states:>5.1f}% | {s.mean:>7.3f}/{s.median:>7.3f}/{s.std:>6.3f}/[{s.minimum:>7.3f},{s.maximum:>7.3f}] | {a.mean:>7.3f}/{a.median:>7.3f}/{a.std:>6.3f} | {report.positive_advantage_pct[i]:>6.2f}%")
    lines += ["", "[2] ENVIRONMENT COUNTERFACTUALS", f"  BUY > HOLD immediate reward:  {report.buy_better_than_hold_pct:.2f}%", f"  SELL > HOLD immediate reward: {report.sell_better_than_hold_pct:.2f}%", f"  policy == best immediate action: {report.policy_immediate_match_pct:.2f}%"]
    lines += ["", "[3] MARKET SIGNAL / 12 FEATURES", "feature | mean | std | range | corr(feature,next_return)"]
    for i, d in enumerate(report.feature_stats):
        lines.append(f"  {i:>2} | {d.mean:>7.3f} | {d.std:>6.3f} | [{d.minimum:>6.3f},{d.maximum:>6.3f}] | {report.feature_next_return_correlation[i]:>8.4f}")
    lines += ["", "[4] SPARSE ENCODER", f"  unique sparse states: {report.sparse_unique_states}", f"  unique encoded vectors: {report.encoded_unique_vectors}", f"  active magnitude mean/std: {report.active_magnitude.mean:.5f}/{report.active_magnitude.std:.5f}"]
    lines += ["", "[5] RECURRENT NETWORK", f"  unique hidden states: {report.hidden_unique_states}", f"  hidden activation mean/std: {report.hidden_activation.mean:.4f}/{report.hidden_activation.std:.4f}", f"  tanh saturation: {report.hidden_saturation_pct:.2f}%", f"  recurrent output effect mean/max: {report.recurrent_effect.mean:.6f}/{report.recurrent_effect.maximum:.6f}", f"  state change mean/std: {report.state_change.mean:.6f}/{report.state_change.std:.6f}", f"  score change mean/std: {report.score_change.mean:.6f}/{report.score_change.std:.6f}", f"  same observation -> multiple hidden states: {report.same_observation_multiple_hidden_states}"]
    lines += ["", "[6] TOPOLOGY / WEIGHTS", f"  input/recurrent/output connections: {report.input_weight and ''}", f"  input weights mean/std/range: {report.input_weight.mean:.4f}/{report.input_weight.std:.4f}/[{report.input_weight.minimum:.4f},{report.input_weight.maximum:.4f}]", f"  recurrent weights mean/std/range: {report.recurrent_weight.mean:.4f}/{report.recurrent_weight.std:.4f}/[{report.recurrent_weight.minimum:.4f},{report.recurrent_weight.maximum:.4f}]", f"  output weights mean/std/range: {report.output_weight.mean:.4f}/{report.output_weight.std:.4f}/[{report.output_weight.minimum:.4f},{report.output_weight.maximum:.4f}]"]
    lines += ["", "[7] GENERALIZATION", f"  unseen HOLD/BUY/SELL: {report.unseen_action_pct[0]:.1f}%/{report.unseen_action_pct[1]:.1f}%/{report.unseen_action_pct[2]:.1f}%", f"  unseen return/drawdown/trades: {report.unseen_return_pct:.3f}%/{report.unseen_drawdown_pct:.3f}%/{report.unseen_trades:.2f}", "  inferred regimes: " + ", ".join(f"{r}={n}" for r, n in report.inferred_regimes)]
    lines += ["", "[8] FINDINGS"] + [f"  - {x}" for x in report.findings]
    lines += ["", "[9] SUSPECTED ROOT CAUSES"] + [f"  {i}. {x}" for i, x in enumerate(report.suspects, 1)]
    return "\n".join(lines)
