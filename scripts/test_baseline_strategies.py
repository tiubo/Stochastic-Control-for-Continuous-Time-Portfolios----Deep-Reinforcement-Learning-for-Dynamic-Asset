"""
Comprehensive Baseline Strategies Testing Script

Tests all baseline strategies on real market data and generates comparison report.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.baselines import (
    MertonStrategy,
    MeanVarianceStrategy,
    EqualWeightStrategy,
    BuyAndHoldStrategy,
    RiskParityStrategy
)


def load_data(data_path: str = 'data/processed/complete_dataset.csv'):
    """Load processed market data."""
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return data


def extract_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Extract return columns from dataset."""
    return_cols = [col for col in data.columns if col.startswith('return_')]
    returns = data[return_cols].copy()
    # Rename columns to remove 'return_' prefix
    returns.columns = [col.replace('return_', '') for col in returns.columns]
    return returns


def run_all_backtests(returns: pd.DataFrame, initial_value: float = 100000.0):
    """Run backtests for all baseline strategies."""

    results = {}

    print("="*80)
    print("BASELINE STRATEGIES BACKTEST")
    print("="*80)
    print(f"Data: {returns.index[0]} to {returns.index[-1]} ({len(returns)} days)")
    print(f"Assets: {', '.join(returns.columns)}")
    print(f"Initial Value: ${initial_value:,.2f}")
    print("="*80)

    # 1. Merton Strategy
    print("\n1. MERTON STRATEGY")
    print("-" * 80)
    merton = MertonStrategy(
        risk_free_rate=0.02,
        estimation_window=252,
        rebalance_freq=20
    )
    results['Merton'] = merton.backtest(
        returns=returns,
        initial_value=initial_value,
        risk_aversion=1.0,
        transaction_cost=0.001
    )
    print(f"Total Return:  {results['Merton']['total_return']:>8.2%}")
    print(f"Sharpe Ratio:  {results['Merton']['sharpe_ratio']:>8.3f}")
    print(f"Max Drawdown:  {results['Merton']['max_drawdown']:>8.2%}")
    print(f"Avg Turnover:  {results['Merton']['avg_turnover']:>8.3f}")
    print(f"Final Value:   ${results['Merton']['final_value']:>12,.2f}")

    # 2. Mean-Variance Optimization
    print("\n2. MEAN-VARIANCE OPTIMIZATION")
    print("-" * 80)
    mv = MeanVarianceStrategy(
        estimation_window=252,
        rebalance_freq=20,
        risk_aversion=2.0,
        allow_short=False
    )
    results['Mean-Variance'] = mv.backtest(
        returns=returns,
        initial_value=initial_value,
        risk_aversion=2.0,
        transaction_cost=0.001
    )
    print(f"Total Return:  {results['Mean-Variance']['total_return']:>8.2%}")
    print(f"Sharpe Ratio:  {results['Mean-Variance']['sharpe_ratio']:>8.3f}")
    print(f"Max Drawdown:  {results['Mean-Variance']['max_drawdown']:>8.2%}")
    print(f"Avg Turnover:  {results['Mean-Variance']['avg_turnover']:>8.3f}")
    print(f"Final Value:   ${results['Mean-Variance']['final_value']:>12,.2f}")

    # 3. Equal-Weight
    print("\n3. EQUAL-WEIGHT (1/N)")
    print("-" * 80)
    eq = EqualWeightStrategy(rebalance_freq=20)
    results['Equal-Weight'] = eq.backtest(
        returns=returns,
        initial_value=initial_value,
        transaction_cost=0.001
    )
    print(f"Total Return:  {results['Equal-Weight']['total_return']:>8.2%}")
    print(f"Sharpe Ratio:  {results['Equal-Weight']['sharpe_ratio']:>8.3f}")
    print(f"Max Drawdown:  {results['Equal-Weight']['max_drawdown']:>8.2%}")
    print(f"Avg Turnover:  {results['Equal-Weight']['avg_turnover']:>8.3f}")
    print(f"Final Value:   ${results['Equal-Weight']['final_value']:>12,.2f}")

    # 4. Buy-and-Hold
    print("\n4. BUY-AND-HOLD")
    print("-" * 80)
    bh = BuyAndHoldStrategy()
    initial_allocation = bh.allocate(len(returns.columns))
    print(f"Initial Allocation: {initial_allocation}")
    results['Buy-and-Hold'] = bh.backtest(
        returns=returns,
        initial_value=initial_value,
        transaction_cost=0.001
    )
    print(f"Total Return:  {results['Buy-and-Hold']['total_return']:>8.2%}")
    print(f"Sharpe Ratio:  {results['Buy-and-Hold']['sharpe_ratio']:>8.3f}")
    print(f"Max Drawdown:  {results['Buy-and-Hold']['max_drawdown']:>8.2%}")
    print(f"Avg Turnover:  {results['Buy-and-Hold']['avg_turnover']:>8.3f}")
    print(f"Final Value:   ${results['Buy-and-Hold']['final_value']:>12,.2f}")

    # 5. Risk Parity
    print("\n5. RISK PARITY")
    print("-" * 80)
    rp = RiskParityStrategy(estimation_window=60, rebalance_freq=20)
    results['Risk Parity'] = rp.backtest(
        returns=returns,
        initial_value=initial_value,
        transaction_cost=0.001
    )
    print(f"Total Return:  {results['Risk Parity']['total_return']:>8.2%}")
    print(f"Sharpe Ratio:  {results['Risk Parity']['sharpe_ratio']:>8.3f}")
    print(f"Max Drawdown:  {results['Risk Parity']['max_drawdown']:>8.2%}")
    print(f"Avg Turnover:  {results['Risk Parity']['avg_turnover']:>8.3f}")
    print(f"Final Value:   ${results['Risk Parity']['final_value']:>12,.2f}")

    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table of all strategies."""

    comparison = pd.DataFrame({
        strategy: [
            res['total_return'],
            res['sharpe_ratio'],
            res['max_drawdown'],
            res['avg_turnover'],
            res['final_value']
        ]
        for strategy, res in results.items()
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Avg Turnover', 'Final Value'])

    return comparison


def plot_comparison(results: dict, save_path: str = None):
    """Plot wealth trajectories comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Wealth Trajectories
    ax = axes[0, 0]
    for strategy, res in results.items():
        values = res['portfolio_values']
        ax.plot(values, label=strategy, linewidth=2, alpha=0.8)
    ax.set_title('Wealth Trajectories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Drawdown Comparison
    ax = axes[0, 1]
    for strategy, res in results.items():
        values = res['portfolio_values']
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax * 100
        ax.plot(drawdown, label=strategy, linewidth=2, alpha=0.8)
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # 3. Performance Metrics Bar Chart
    ax = axes[1, 0]
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
    x = np.arange(len(results))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = []
        for res in results.values():
            if metric == 'Total Return':
                values.append(res['total_return'] * 100)
            elif metric == 'Sharpe Ratio':
                values.append(res['sharpe_ratio'])
            elif metric == 'Max Drawdown':
                values.append(res['max_drawdown'] * 100)
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)

    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results.keys(), rotation=15, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Risk-Return Scatter
    ax = axes[1, 1]
    for strategy, res in results.items():
        returns_data = res['portfolio_returns']
        annual_return = res['total_return'] * 100
        annual_vol = returns_data.std() * np.sqrt(252) * 100
        ax.scatter(annual_vol, annual_return, s=200, alpha=0.7, label=strategy)
        ax.annotate(strategy, (annual_vol, annual_return),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Total Return (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


def main():
    """Main execution function."""

    # Load data
    print("Loading data...")
    data = load_data()
    returns = extract_returns(data)

    print(f"Loaded {len(returns)} days of data")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"Assets: {', '.join(returns.columns)}")

    # Run backtests
    print("\nRunning backtests...")
    results = run_all_backtests(returns, initial_value=100000.0)

    # Create comparison table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    comparison = create_comparison_table(results)
    print(comparison.to_string())

    # Save results
    output_dir = Path('simulations/baseline_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison.to_csv(output_dir / 'baseline_comparison.csv')
    print(f"\nComparison table saved to: {output_dir / 'baseline_comparison.csv'}")

    # Plot and save
    fig = plot_comparison(results, save_path=output_dir / 'baseline_comparison.png')

    # Save detailed results
    for strategy, res in results.items():
        strategy_dir = output_dir / strategy.replace(' ', '_').replace('-', '_').lower()
        strategy_dir.mkdir(exist_ok=True)

        # Save portfolio values
        pd.DataFrame({
            'portfolio_value': res['portfolio_values']
        }).to_csv(strategy_dir / 'portfolio_values.csv')

        # Save weights history
        if res['portfolio_weights']:
            weights_df = pd.DataFrame(
                res['portfolio_weights'],
                columns=returns.columns
            )
            weights_df.to_csv(strategy_dir / 'weights_history.csv')

    print(f"\nDetailed results saved to: {output_dir}")

    # Identify best strategies
    print("\n" + "="*80)
    print("BEST STRATEGIES BY METRIC")
    print("="*80)

    best_return = comparison.loc['Total Return'].idxmax()
    best_sharpe = comparison.loc['Sharpe Ratio'].idxmax()
    best_drawdown = comparison.loc['Max Drawdown'].idxmin()

    print(f"Best Total Return:  {best_return} ({comparison.loc['Total Return', best_return]:.2%})")
    print(f"Best Sharpe Ratio:  {best_sharpe} ({comparison.loc['Sharpe Ratio', best_sharpe]:.3f})")
    print(f"Best Drawdown:      {best_drawdown} ({comparison.loc['Max Drawdown', best_drawdown]:.2%})")

    print("\n" + "="*80)
    print("BASELINE STRATEGIES TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
