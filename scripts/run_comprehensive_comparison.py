"""
Comprehensive Strategy Comparison: Baselines vs RL Agents

Backtests all 8 strategies on the same data:
- Baselines: Merton, Mean-Variance, Equal-Weight, Buy-and-Hold, Risk Parity
- RL Agents: DQN, PPO, SAC

Generates:
- Performance comparison table
- Multi-strategy equity curves
- Risk-return scatter plot
- Detailed per-strategy results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    BacktestResults,
    # Baseline adapters
    MertonStrategyAdapter,
    MeanVarianceAdapter,
    EqualWeightAdapter,
    BuyAndHoldAdapter,
    RiskParityAdapter,
    # RL adapters
    DQNStrategyAdapter,
    PPOStrategyAdapter,
    SACStrategyAdapter
)


def load_data(data_path: str) -> tuple:
    """Load and prepare market data."""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Separate prices and returns
    price_cols = [col for col in data.columns if col.startswith('price_')]
    return_cols = [col for col in data.columns if col.startswith('return_')]

    returns = data[return_cols].copy()

    print(f"  Data shape: {data.shape}")
    print(f"  Assets: {len(price_cols)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    return data, returns


def create_baseline_strategies(n_assets: int) -> Dict:
    """Create all baseline strategy adapters."""
    return {
        'Merton': MertonStrategyAdapter(risk_aversion=2.0),
        'Mean-Variance': MeanVarianceAdapter(risk_aversion=2.0, lookback=252),
        'Equal-Weight': EqualWeightAdapter(n_assets=n_assets),
        'Buy-and-Hold': BuyAndHoldAdapter(n_assets=n_assets),
        'Risk-Parity': RiskParityAdapter(n_assets=n_assets, lookback=60)
    }


def create_rl_strategies(models_dir: str = 'models', device: str = 'cpu') -> Dict:
    """
    Create RL strategy adapters from trained models.

    Args:
        models_dir: Directory containing trained models
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary of RL strategy adapters, or empty dict if models not found
    """
    strategies = {}
    models_path = Path(models_dir)

    # DQN
    dqn_path = models_path / 'dqn_trained.pth'
    if dqn_path.exists():
        print(f"  ✓ Found DQN model: {dqn_path}")
        strategies['DQN'] = DQNStrategyAdapter(str(dqn_path), device=device)
    else:
        print(f"  ✗ DQN model not found: {dqn_path}")

    # PPO
    ppo_path = models_path / 'ppo' / 'ppo_final.pth'
    if ppo_path.exists():
        print(f"  ✓ Found PPO model: {ppo_path}")
        strategies['PPO'] = PPOStrategyAdapter(str(ppo_path), device=device)
    else:
        print(f"  ✗ PPO model not found: {ppo_path}")

    # SAC
    sac_path = models_path / 'sac_trained.pth'
    if sac_path.exists():
        print(f"  ✓ Found SAC model: {sac_path}")
        strategies['SAC'] = SACStrategyAdapter(str(sac_path), device=device)
    else:
        print(f"  ✗ SAC model not found: {sac_path}")

    return strategies


def run_backtests(
    strategies: Dict,
    data: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig
) -> Dict[str, BacktestResults]:
    """Run backtests for all strategies."""
    results = {}
    engine = BacktestEngine(config)

    for name, strategy in strategies.items():
        print(f"\nBacktesting {name}...")
        try:
            result = engine.run(strategy, data, returns)
            results[name] = result
            print(f"  Final portfolio value: ${result.final_value:,.2f}")
            print(f"  Total return: {result.total_return:.2%}")
            print(f"  Sharpe ratio: {result.sharpe_ratio:.3f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def create_comparison_table(results: Dict[str, BacktestResults]) -> pd.DataFrame:
    """Create performance comparison table."""
    comparison = []

    for name, result in results.items():
        comparison.append({
            'Strategy': name,
            'Total Return (%)': result.total_return * 100,
            'Annual Return (%)': result.annual_return * 100,
            'Volatility (%)': result.volatility * 100,
            'Sharpe Ratio': result.sharpe_ratio,
            'Sortino Ratio': result.sortino_ratio,
            'Calmar Ratio': result.calmar_ratio,
            'Max Drawdown (%)': result.max_drawdown * 100,
            'Win Rate (%)': result.win_rate * 100,
            'Avg Turnover (%)': result.avg_turnover * 100,
            'Total Trades': result.total_trades
        })

    df = pd.DataFrame(comparison)

    # Sort by Sharpe ratio (descending)
    df = df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)

    return df


def plot_comprehensive_comparison(
    results: Dict[str, BacktestResults],
    output_path: str
):
    """
    Create comprehensive 4-panel visualization.

    Panels:
    1. Equity curves (all strategies)
    2. Risk-return scatter
    3. Drawdown comparison
    4. Performance metrics heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Equity Curves
    ax = axes[0, 0]
    for name, result in results.items():
        # Determine color based on strategy type
        if name in ['DQN', 'PPO', 'SAC']:
            linestyle = '-'
            linewidth = 2.5
        else:
            linestyle = '--'
            linewidth = 1.5

        ax.plot(result.portfolio_values, label=name, linestyle=linestyle, linewidth=linewidth)

    ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see all strategies clearly

    # Panel 2: Risk-Return Scatter
    ax = axes[0, 1]
    annual_returns = [r.annual_return * 100 for r in results.values()]
    volatilities = [r.volatility * 100 for r in results.values()]
    names = list(results.keys())

    # Color by strategy type
    colors = ['red' if name in ['DQN', 'PPO', 'SAC'] else 'blue' for name in names]

    scatter = ax.scatter(volatilities, annual_returns, c=colors, s=200, alpha=0.6, edgecolors='black')

    for i, name in enumerate(names):
        ax.annotate(name, (volatilities[i], annual_returns[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Volatility (% p.a.)')
    ax.set_ylabel('Annual Return (%)')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='RL Agents'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Baselines')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Panel 3: Maximum Drawdown Comparison
    ax = axes[1, 0]
    dd_values = [r.max_drawdown * 100 for r in results.values()]
    colors_dd = ['red' if name in ['DQN', 'PPO', 'SAC'] else 'blue' for name in names]

    bars = ax.barh(names, dd_values, color=colors_dd, alpha=0.6, edgecolor='black')
    ax.set_title('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Max Drawdown (%)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()  # Lower drawdown is better

    # Panel 4: Performance Metrics Heatmap
    ax = axes[1, 1]

    # Create normalized metrics for heatmap
    metrics_df = pd.DataFrame({
        'Total Return': [r.total_return for r in results.values()],
        'Sharpe Ratio': [r.sharpe_ratio for r in results.values()],
        'Sortino Ratio': [r.sortino_ratio for r in results.values()],
        'Calmar Ratio': [r.calmar_ratio for r in results.values()],
        'Win Rate': [r.win_rate for r in results.values()],
    }, index=names)

    # Normalize each metric to [0, 1] for heatmap
    metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

    sns.heatmap(metrics_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Score'}, ax=ax, linewidths=0.5)
    ax.set_title('Performance Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive strategy comparison')
    parser.add_argument('--data', type=str,
                       default='data/processed/complete_dataset.csv',
                       help='Path to market data CSV')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained RL models')
    parser.add_argument('--output-dir', type=str,
                       default='simulations/comprehensive_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for RL model inference')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial portfolio value')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost (e.g., 0.001 = 0.1%)')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Run only baseline strategies (skip RL)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 80)

    # Load data
    data, returns = load_data(args.data)
    n_assets = len([col for col in data.columns if col.startswith('return_')])

    # Create backtest config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        slippage=0.0005,
        rebalance_frequency=1
    )

    print(f"\nBacktest Configuration:")
    print(f"  Initial capital: ${config.initial_capital:,.2f}")
    print(f"  Transaction cost: {config.transaction_cost:.2%}")
    print(f"  Slippage: {config.slippage:.2%}")
    print(f"  Rebalance frequency: {config.rebalance_frequency} days")

    # Create strategies
    print(f"\nCreating strategies...")
    print(f"\nBaseline Strategies:")
    all_strategies = create_baseline_strategies(n_assets)
    print(f"  ✓ Created {len(all_strategies)} baseline strategies")

    if not args.baselines_only:
        print(f"\nRL Strategies:")
        rl_strategies = create_rl_strategies(args.models_dir, args.device)
        all_strategies.update(rl_strategies)
        if rl_strategies:
            print(f"  ✓ Created {len(rl_strategies)} RL strategies")
        else:
            print(f"  ⚠ No RL models found. Run training first or use --baselines-only")

    print(f"\nTotal strategies: {len(all_strategies)}")

    # Run backtests
    print(f"\n" + "=" * 80)
    print("RUNNING BACKTESTS")
    print("=" * 80)

    results = run_backtests(all_strategies, data, returns, config)

    if not results:
        print("\n❌ No successful backtests. Exiting.")
        return

    # Create comparison table
    print(f"\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    comparison_df = create_comparison_table(results)
    print(f"\n{comparison_df.to_string(index=False)}")

    # Save comparison table
    comparison_path = output_dir / 'comprehensive_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Saved comparison table: {comparison_path}")

    # Create visualization
    plot_path = output_dir / 'comprehensive_comparison.png'
    plot_comprehensive_comparison(results, str(plot_path))

    # Save detailed results for each strategy
    for name, result in results.items():
        strategy_dir = output_dir / name.lower().replace('-', '_').replace(' ', '_')
        strategy_dir.mkdir(exist_ok=True)

        # Portfolio values
        pd.DataFrame({
            'portfolio_value': result.portfolio_values
        }).to_csv(strategy_dir / 'portfolio_values.csv')

        # Weights history
        weights_df = pd.DataFrame(
            result.weights_history,
            columns=[f'asset_{i}' for i in range(len(result.weights_history[0]))]
        )
        weights_df.to_csv(strategy_dir / 'weights_history.csv')

        # Trades
        if result.trades:
            trades_df = pd.DataFrame([
                {'step': t.step, 'action': t.action, 'cost': t.cost}
                for t in result.trades
            ])
            trades_df.to_csv(strategy_dir / 'trades.csv', index=False)

    print(f"\n✓ Saved detailed results for {len(results)} strategies")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Backtested {len(results)} strategies successfully")
    print(f"✅ Best Sharpe ratio: {comparison_df.iloc[0]['Strategy']} ({comparison_df.iloc[0]['Sharpe Ratio']:.3f})")
    print(f"✅ Best total return: {comparison_df.loc[comparison_df['Total Return (%)'].idxmax(), 'Strategy']} "
          f"({comparison_df['Total Return (%)'].max():.2f}%)")
    print(f"✅ Lowest drawdown: {comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin(), 'Strategy']} "
          f"({comparison_df['Max Drawdown (%)'].min():.2f}%)")

    print(f"\n📁 All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
