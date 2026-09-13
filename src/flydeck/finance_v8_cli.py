from __future__ import annotations

from .agent import Agent
from .finance import CryptoTradingEnvironment
from .finance_encoder import SparseMarketEncoder
from .finance_training import evaluate_buy_and_hold, evaluate_hold
from .finance_v8 import train_synthetic_crypto_v8


def main() -> None:
    market_length = 256
    max_steps = 200
    training_seed = 100
    evaluation_seed = 10_000

    environment = CryptoTradingEnvironment((), max_steps=max_steps)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    agent = Agent(
        observation_size=encoder.output_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )

    print("FlyDeck Agent - Finance V8")
    print("learning: sparse k-WTA + recurrent circuit + TD(lambda)")
    print("V8 experiment: selected-action-only output TD update")
    print("competitor output weights are not directly pushed down")
    print(f"market features: {environment.observation_size}")
    print(f"sparse state: {encoder.output_size} units | {encoder.active_units} active ({encoder.sparsity:.1%})")
    print("actions: HOLD / BUY / SELL")
    print(f"connections: {agent.network.connection_count}")
    print()

    training = train_synthetic_crypto_v8(
        agent,
        episodes=100,
        market_length=market_length,
        max_steps=max_steps,
        seed=training_seed,
        epsilon=0.30,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        trade_penalty=0.0025,
        invalid_action_penalty=0.001,
        drawdown_penalty=0.02,
        discount=0.97,
        trace_decay=0.85,
    )

    total_actions = training.total_actions
    print("Training:")
    print(f"episodes: {training.episodes}")
    print(f"average return: {training.average_return_pct:.3f}%")
    print(f"best return: {training.best_return_pct:.3f}%")
    print(f"last return: {training.last_return_pct:.3f}%")
    print(f"average drawdown: {training.average_drawdown_pct:.3f}%")
    print(f"trades: {training.total_trades}")
    print(f"trade frequency: {training.trade_frequency:.3f}")
    print(f"average prediction error: {training.average_prediction_error:.5f}")
    print("policy actions:")
    print(f"  HOLD: {training.hold_actions} ({training.hold_actions / max(1, total_actions):.1%})")
    print(f"  BUY:  {training.buy_actions} ({training.buy_actions / max(1, total_actions):.1%})")
    print(f"  SELL: {training.sell_actions} ({training.sell_actions / max(1, total_actions):.1%})")

    evaluation = _evaluate_unseen(agent, evaluation_seed, market_length, max_steps)
    hold = evaluate_hold(seed=evaluation_seed, market_length=market_length, max_steps=max_steps)
    buy_hold = evaluate_buy_and_hold(seed=evaluation_seed, market_length=market_length, max_steps=max_steps)

    print()
    print("Unseen-market evaluation:")
    print(f"agent return: {evaluation.return_pct:.3f}%")
    print(f"agent portfolio: {evaluation.final_portfolio:.2f}")
    print(f"agent drawdown: {evaluation.drawdown_pct:.3f}%")
    print(f"agent trades: {evaluation.trades}")
    print(f"agent trade frequency: {evaluation.trades / max(1, max_steps):.3f}")
    print("policy actions:")
    print(f"  HOLD: {evaluation.counts[0]} ({evaluation.counts[0] / max(1, max_steps):.1%})")
    print(f"  BUY:  {evaluation.counts[1]} ({evaluation.counts[1] / max(1, max_steps):.1%})")
    print(f"  SELL: {evaluation.counts[2]} ({evaluation.counts[2] / max(1, max_steps):.1%})")
    print("score diagnostics:")
    print(f"  HOLD mean: {evaluation.score_means[0]:.4f}")
    print(f"  BUY mean:  {evaluation.score_means[1]:.4f}")
    print(f"  SELL mean: {evaluation.score_means[2]:.4f}")
    print(f"  BUY-HOLD advantage mean: {evaluation.score_advantages[0]:.4f}")
    print(f"  SELL-HOLD advantage mean: {evaluation.score_advantages[1]:.4f}")

    print()
    print("Baselines on the same unseen market:")
    print(f"  HOLD:      {hold.return_pct:.3f}% | drawdown {hold.max_drawdown_pct:.3f}% | trades {hold.trades}")
    print(f"  BUY&HOLD:  {buy_hold.return_pct:.3f}% | drawdown {buy_hold.max_drawdown_pct:.3f}% | trades {buy_hold.trades}")
    print()
    print(f"connections: {agent.network.connection_count}")
    print(f"memory: {len(agent.memory)}")


def _evaluate_unseen(agent: Agent, seed: int, market_length: int, max_steps: int):
    from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket

    environment = CryptoTradingEnvironment(
        SyntheticCryptoMarket(length=market_length, seed=seed).generate(),
        max_steps=max_steps,
        trade_penalty=0.0025,
        invalid_action_penalty=0.001,
        drawdown_penalty=0.02,
    )
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    observation = environment.reset()
    agent.network.reset()
    encoder.reset()
    scores = agent.observe(encoder.encode(observation))
    counts = [0, 0, 0]
    score_totals = [0.0, 0.0, 0.0]
    score_advantages = [[], []]
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent.choose_action(scores)
        counts[action] += 1
        for index in range(3):
            score_totals[index] += scores[index]
        score_advantages[0].append(scores[1] - scores[0])
        score_advantages[1].append(scores[2] - scores[0])

        result = environment.step(action)
        encoder.observe_action(action)
        total_reward += result.reward
        if result.done:
            break
        scores = agent.observe(encoder.encode(result.observation))

    metrics = environment.episode_result(total_reward)

    class Evaluation:
        return_pct = metrics.return_pct
        final_portfolio = metrics.final_portfolio
        drawdown_pct = metrics.max_drawdown_pct
        trades = metrics.trades
        counts = counts
        score_means = tuple(value / max(1, sum(counts)) for value in score_totals)
        score_advantages = (
            sum(score_advantages[0]) / max(1, len(score_advantages[0])),
            sum(score_advantages[1]) / max(1, len(score_advantages[1])),
        )

    return Evaluation


if __name__ == "__main__":
    main()
