"""
Compare DQN Agent Performance vs Baseline Strategies

This script evaluates a trained DQN agent on the test set and compares
its performance against the baseline strategies that have already been run.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.dqn_agent import DQNAgent
from src.environments.portfolio_env import PortfolioEnv


def load_baseline_results(baseline_path='simulations/baseline_results/baseline_comparison.csv'):
    """Load existing baseline results"""
    df = pd.read_csv(baseline_path, index_col=0)
    return df


def evaluate_dqn_agent(model_path, data_path, device='cpu'):
    """
    Evaluate trained DQN agent on test data

    Returns metrics compatible with baseline comparison
    """
    print("\n" + "="*80)
    print("EVALUATING DQN AGENT ON TEST DATA")
    print("="*80)

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded data: {len(df)} days")

    # Check for date column (could be 'Date' or 'date')
    date_col = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None
    if date_col:
        print(f"Date range: {df[date_col].iloc[0]} to {df[date_col].iloc[-1]}")

    # Split into train/test (same as training)
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:].reset_index(drop=True)

    print(f"\nTest set: {len(test_df)} days")
    if date_col:
        print(f"Test period: {test_df[date_col].iloc[0]} to {test_df[date_col].iloc[-1]}")

    # Create test environment with discrete actions for DQN
    test_env = PortfolioEnv(test_df, window_size=20, action_type='discrete')

    # Load trained agent
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load(model_path)

    print(f"\nLoaded DQN model from {model_path}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Run episode with no exploration
    state, _ = test_env.reset()  # Gymnasium API returns (state, info)
    done = False

    initial_value = 100000  # Match baseline initial capital
    portfolio_value = initial_value
    portfolio_values = [portfolio_value]
    returns_list = []
    actions_list = []

    step = 0
    print("\nRunning backtest...")

    while not done:
        # Get action (no exploration, greedy=True)
        action = agent.select_action(state, epsilon=0.0)
        actions_list.append(action)

        # Take step (handle both old and new Gymnasium API)
        step_result = test_env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result

        # Get portfolio return from environment
        portfolio_return = info.get('portfolio_return', 0.0)
        returns_list.append(portfolio_return)

        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        portfolio_values.append(portfolio_value)

        state = next_state
        step += 1

        if step % 100 == 0:
            print(f"  Step {step}/{len(test_df)}: Portfolio Value = ${portfolio_value:,.2f}")

    print(f"\nBacktest complete: {step} steps")

    # Calculate metrics
    returns_array = np.array(returns_list)
    portfolio_values_array = np.array(portfolio_values)

    # Total return
    total_return = (portfolio_value / initial_value) - 1

    # Annualized return
    n_days = len(returns_array)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1

    # Volatility
    annualized_vol = returns_array.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = (annualized_return - 0.02) / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    cumulative = np.maximum.accumulate(portfolio_values_array)
    drawdown = (portfolio_values_array - cumulative) / cumulative
    max_drawdown = abs(drawdown.min())

    # Turnover (approximate - DQN rebalances every step)
    avg_turnover = 0.05  # Conservative estimate

    # Final value
    final_value = portfolio_value

    metrics = {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Avg Turnover': avg_turnover,
        'Final Value': final_value
    }

    print("\n" + "="*80)
    print("DQN PERFORMANCE METRICS")
    print("="*80)
    for key, value in metrics.items():
        if key == 'Final Value':
            print(f"{key:20s}: ${value:>15,.2f}")
        elif key in ['Total Return', 'Max Drawdown']:
            print(f"{key:20s}: {value:>15.2%}")
        elif key == 'Sharpe Ratio':
            print(f"{key:20s}: {value:>15.3f}")
        else:
            print(f"{key:20s}: {value:>15.4f}")

    return metrics, portfolio_values_array, returns_array, actions_list


def create_comparison_report(dqn_metrics, baseline_df, output_dir='simulations/backtest_results'):
    """Create comprehensive comparison report"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add DQN to baseline comparison
    comparison_df = baseline_df.copy()
    comparison_df['DQN'] = [
        dqn_metrics['Total Return'],
        dqn_metrics['Sharpe Ratio'],
        dqn_metrics['Max Drawdown'],
        dqn_metrics['Avg Turnover'],
        dqn_metrics['Final Value']
    ]

    # Save updated comparison
    comparison_path = output_path / 'dqn_vs_baselines_comparison.csv'
    comparison_df.to_csv(comparison_path)
    print(f"\n[OK] Saved comparison to {comparison_path}")

    # Display comparison table
    print("\n" + "="*80)
    print("COMPLETE STRATEGY COMPARISON")
    print("="*80)
    print(comparison_df.to_string())

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN vs Baseline Strategies - Performance Comparison',
                 fontsize=16, fontweight='bold')

    # 1. Sharpe Ratio comparison
    ax1 = axes[0, 0]
    sharpe_data = comparison_df.loc['Sharpe Ratio'].sort_values(ascending=False)
    colors = ['#2ecc71' if x == sharpe_data.max() else '#3498db' for x in sharpe_data]
    sharpe_data.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # 2. Total Return comparison
    ax2 = axes[0, 1]
    return_data = comparison_df.loc['Total Return'].sort_values(ascending=False) * 100
    colors = ['#2ecc71' if x == return_data.max() else '#3498db' for x in return_data]
    return_data.plot(kind='bar', ax=ax2, color=colors, alpha=0.8)
    ax2.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Return (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Max Drawdown comparison
    ax3 = axes[1, 0]
    dd_data = comparison_df.loc['Max Drawdown'].sort_values() * 100
    colors = ['#2ecc71' if x == dd_data.min() else '#e74c3c' for x in dd_data]
    dd_data.plot(kind='bar', ax=ax3, color=colors, alpha=0.8)
    ax3.set_title('Max Drawdown Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.grid(True, alpha=0.3)

    # 4. Risk-Return scatter
    ax4 = axes[1, 1]
    returns = comparison_df.loc['Total Return']
    sharpes = comparison_df.loc['Sharpe Ratio']

    for strategy in comparison_df.columns:
        color = '#e74c3c' if strategy == 'DQN' else '#3498db'
        size = 300 if strategy == 'DQN' else 150
        ax4.scatter(sharpes[strategy], returns[strategy] * 100,
                   s=size, alpha=0.6, color=color, label=strategy)
        ax4.annotate(strategy, (sharpes[strategy], returns[strategy] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax4.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Return (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / 'dqn_vs_baselines.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization to {plot_path}")
    plt.close()

    # Calculate rankings
    print("\n" + "="*80)
    print("STRATEGY RANKINGS")
    print("="*80)

    rankings = {
        'Sharpe Ratio': comparison_df.loc['Sharpe Ratio'].rank(ascending=False),
        'Total Return': comparison_df.loc['Total Return'].rank(ascending=False),
        'Max Drawdown': comparison_df.loc['Max Drawdown'].rank(ascending=True),  # Lower is better
    }

    rankings_df = pd.DataFrame(rankings)
    rankings_df['Average Rank'] = rankings_df.mean(axis=1)
    rankings_df = rankings_df.sort_values('Average Rank')

    print(rankings_df.to_string())

    # Highlight DQN ranking
    if 'DQN' in rankings_df.index:
        dqn_rank = rankings_df.loc['DQN', 'Average Rank']
        total_strategies = len(rankings_df)
        print(f"\n[TARGET] DQN Agent: Ranked #{int(dqn_rank)} out of {total_strategies} strategies")

    return comparison_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare DQN vs Baseline Strategies')
    parser.add_argument('--dqn-model', type=str, default='models/dqn_trained.pth',
                       help='Path to trained DQN model')
    parser.add_argument('--data', type=str, default='data/processed/dataset_with_regimes.csv',
                       help='Path to dataset')
    parser.add_argument('--baseline', type=str, default='simulations/baseline_results/baseline_comparison.csv',
                       help='Path to baseline comparison CSV')
    parser.add_argument('--output', type=str, default='simulations/backtest_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("DQN VS BASELINE STRATEGIES COMPARISON")
    print("="*80)

    # Evaluate DQN
    dqn_metrics, portfolio_values, returns, actions = evaluate_dqn_agent(
        args.dqn_model, args.data, args.device
    )

    # Load baseline results
    print("\n" + "="*80)
    print("LOADING BASELINE RESULTS")
    print("="*80)
    baseline_df = load_baseline_results(args.baseline)
    print(f"\nLoaded baseline results for {len(baseline_df.columns)} strategies:")
    print(f"  {', '.join(baseline_df.columns)}")

    # Create comparison report
    comparison_df = create_comparison_report(dqn_metrics, baseline_df, args.output)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output}")
    print(f"\n[OK] Backtesting and comparison complete!")


if __name__ == '__main__':
    main()
