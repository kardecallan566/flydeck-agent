from __future__ import annotations

from .agent import Agent
from .environment import CounterEnvironment
from .finance import CryptoTradingEnvironment, SyntheticCryptoMarket
from .finance_encoder import SparseMarketEncoder
from .finance_training import evaluate_buy_and_hold, evaluate_hold
from .finance_v7 import evaluate_synthetic_crypto_v7, train_synthetic_crypto_v7
from .finance_v7_diagnostics import aggregate_diagnostics, diagnose_policy, format_diagnostics
from .navigation import GridNavigationEnvironment


def main() -> None:
    environment = CryptoTradingEnvironment(SyntheticCryptoMarket(length=256, seed=42).generate(), max_steps=200)
    encoder = SparseMarketEncoder(feature_count=environment.observation_size, winners=4)
    agent = Agent(
        observation_size=encoder.output_size,
        action_size=environment.action_size,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )

    opportunity_margin = 0.08
    confidence_threshold = 0.55
    print("FlyDeck Agent - Synthetic Crypto V7.2")
    print("learning: sparse k-WTA + recurrent circuit + TD(lambda) + raw-action learning")
    print("decision: calibrated value confidence + execution-only opportunity gate")
    print("architecture: train raw policy -> evaluate raw policy -> gate execution only")
    print("diagnostics: scores + sparse-state diversity + inferred regimes")
    print(f"market features: {environment.observation_size}")
    print(f"sparse state: {encoder.output_size} units | {encoder.active_units} active ({encoder.sparsity:.1%})")
    print("actions: HOLD / BUY / SELL")
    print(f"max opportunity margin: {opportunity_margin:.3f}")
    print(f"confidence threshold: {confidence_threshold:.2f}")
    print(f"connections: {agent.network.connection_count}")
    print()

    training = train_synthetic_crypto_v7(
        agent,
        episodes=100,
        market_length=256,
        max_steps=200,
        seed=100,
        epsilon=0.30,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        trade_penalty=0.0025,
        invalid_action_penalty=0.001,
        drawdown_penalty=0.02,
        discount=0.97,
        trace_decay=0.85,
        opportunity_margin=opportunity_margin,
        confidence_threshold=confidence_threshold,
    )

    print(f"episodes: {training.episodes}")
    print(f"average return: {training.average_return_pct:.3f}%")
    print(f"best return: {training.best_return_pct:.3f}%")
    print(f"last return: {training.last_return_pct:.3f}%")
    print(f"average drawdown: {training.average_drawdown_pct:.3f}%")
    print(f"trades: {training.total_trades}")
    print(f"trade frequency: {training.trade_frequency:.3f}")
    print(f"gate activation rate: {training.gate_activation_rate:.1%}")
    print(f"opportunity rate: {training.opportunity_rate:.1%}")
    print(f"average confidence: {training.average_confidence:.3f}")
    print(f"median confidence: {training.median_confidence:.3f}")
    print(f"average prediction error: {training.average_prediction_error:.5f}")
    print("raw policy actions:")
    print(f"  HOLD: {training.raw_hold_actions}")
    print(f"  BUY:  {training.raw_buy_actions}")
    print(f"  SELL: {training.raw_sell_actions}")
    print("executed actions:")
    print(f"  HOLD: {training.executed_hold_actions}")
    print(f"  BUY:  {training.executed_buy_actions}")
    print(f"  SELL: {training.executed_sell_actions}")

    evaluation_seed = 10_000
    evaluation = evaluate_synthetic_crypto_v7(
        agent,
        seed=evaluation_seed,
        market_length=256,
        max_steps=200,
        trade_penalty=0.0025,
        invalid_action_penalty=0.001,
        drawdown_penalty=0.02,
        opportunity_margin=opportunity_margin,
        confidence_threshold=confidence_threshold,
    )
    hold = evaluate_hold(seed=evaluation_seed, market_length=256, max_steps=200)
    buy_hold = evaluate_buy_and_hold(seed=evaluation_seed, market_length=256, max_steps=200)

    print()
    print("Unseen-market evaluation:")
    print(f"agent return: {evaluation.return_pct:.3f}%")
    print(f"agent portfolio: {evaluation.final_portfolio:.2f}")
    print(f"agent drawdown: {evaluation.max_drawdown_pct:.3f}%")
    print(f"agent trades: {evaluation.trades}")
    print(f"agent trade frequency: {evaluation.trade_frequency:.3f}")
    print(f"gate activation rate: {evaluation.gate_activation_rate:.1%}")
    print(f"opportunity rate: {evaluation.opportunity_rate:.1%}")
    print(f"average confidence: {evaluation.average_confidence:.3f}")
    print(f"median confidence: {evaluation.median_confidence:.3f}")
    print(f"average prediction error: {evaluation.average_prediction_error:.5f}")
    print("raw policy actions:")
    print(f"  HOLD: {evaluation.raw_hold_actions}")
    print(f"  BUY:  {evaluation.raw_buy_actions}")
    print(f"  SELL: {evaluation.raw_sell_actions}")
    print("executed actions:")
    print(f"  HOLD: {evaluation.executed_hold_actions}")
    print(f"  BUY:  {evaluation.executed_buy_actions}")
    print(f"  SELL: {evaluation.executed_sell_actions}")
    print()
    print("Baselines on the same unseen market:")
    print(f"  HOLD:      {hold.return_pct:.3f}% | drawdown {hold.max_drawdown_pct:.3f}% | trades {hold.trades}")
    print(f"  BUY&HOLD:  {buy_hold.return_pct:.3f}% | drawdown {buy_hold.max_drawdown_pct:.3f}% | trades {buy_hold.trades}")

    training_diagnostics = [
        diagnose_policy(
            agent,
            SyntheticCryptoMarket(length=256, seed=seed).generate(),
            label=f"training market seed {seed}",
            max_steps=200,
        )
        for seed in range(100, 105)
    ]
    training_summary = aggregate_diagnostics(training_diagnostics, "Training-distribution diagnostics (5 markets)")
    unseen_summary = diagnose_policy(
        agent,
        SyntheticCryptoMarket(length=256, seed=evaluation_seed).generate(),
        label="Unseen-market diagnostics",
        max_steps=200,
    )

    print()
    print("=== V7.2 POLICY DIAGNOSTICS ===")
    print(format_diagnostics(training_summary))
    print()
    print(format_diagnostics(unseen_summary))
    print()
    print(f"connections: {agent.network.connection_count}")
    print(f"memory: {len(agent.memory)}")


def navigation_demo() -> None:
    environment = GridNavigationEnvironment(width=6, height=6, start=(0, 0), goal=(5, 5), max_steps=60)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size)
    result = agent.train(environment, episodes=500, max_steps=60, epsilon=0.35, epsilon_decay=0.995, min_epsilon=0.03)
    print(f"navigation average reward: {result.average_reward:.3f}")


def counter_demo() -> None:
    environment = CounterEnvironment(target=10, max_steps=32)
    agent = Agent(observation_size=environment.observation_size, action_size=environment.action_size)
    result = agent.run(environment)
    print(f"steps: {result.steps}")
    print(f"reward: {result.total_reward:.3f}")


if __name__ == "__main__":
    main()
