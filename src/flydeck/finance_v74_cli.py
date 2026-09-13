from __future__ import annotations

from .agent import Agent
from .finance import CryptoTradingEnvironment
from .finance_encoder import SparseMarketEncoder
from .finance_v7 import train_synthetic_crypto_v7
from .finance_v74_diagnostics import diagnose_finance_agent, format_diagnostic_report


def main() -> None:
    env = CryptoTradingEnvironment([], max_steps=1) if False else None
    encoder = SparseMarketEncoder(feature_count=12, winners=4)
    agent = Agent(
        observation_size=encoder.output_size,
        action_size=3,
        hidden_size=32,
        density=0.10,
        learning_rate=0.005,
        seed=42,
    )
    print("FlyDeck Agent - V7.4 Comprehensive Diagnostic")
    print("training one controlled baseline before auditing all suspected failure layers...")
    train_synthetic_crypto_v7(
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
        opportunity_margin=0.08,
        confidence_threshold=0.55,
    )
    report = diagnose_finance_agent(agent)
    print(format_diagnostic_report(report))


if __name__ == "__main__":
    main()
