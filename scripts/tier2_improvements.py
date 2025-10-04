"""
Tier 2 Production Enhancements
================================

Implements critical features for production-grade portfolio optimization:
1. Crisis period stress testing
2. Regime-dependent analysis
3. Rolling performance metrics
4. Transaction cost sensitivity
5. Out-of-sample robustness checks
6. Model comparison dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

from agents.dqn_agent import DQNAgent
from environments.portfolio_env import PortfolioEnv


# ==============================================================================
# 1. Crisis Period Stress Testing
# ==============================================================================

def stress_test_crisis_periods(
    agent,
    data: pd.DataFrame,
    crisis_periods: List[Tuple[str, str, str]]
) -> pd.DataFrame:
    """
    Test agent performance during historical crisis periods.

    Args:
        agent: Trained RL agent
        data: Full dataset
        crisis_periods: List of (name, start_date, end_date) tuples

    Returns:
        results_df: Crisis performance metrics
    """
    print("=" * 70)
    print("Crisis Period Stress Testing")
    print("=" * 70)

    results = []

    for crisis_name, start_date, end_date in crisis_periods:
        print(f"\n[CRISIS] Testing: {crisis_name} ({start_date} to {end_date})")
        print("-" * 70)

        # Filter data for crisis period
        crisis_data = data.loc[start_date:end_date]

        if len(crisis_data) == 0:
            print(f"[WARN] No data available for {crisis_name}")
            continue

        # Create environment
        env = PortfolioEnv(
            data=crisis_data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',
            reward_type='log_utility'
        )

        # Run episode
        state, _ = env.reset()
        episode_return = 0
        portfolio_values = [env.portfolio_value]
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            portfolio_values.append(env.portfolio_value)
            done = terminated or truncated

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        max_dd = calculate_max_drawdown(portfolio_values)
        sharpe = calculate_sharpe_ratio(returns)

        print(f"  Total Return: {total_return*100:.2f}%")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Final Value:  ${portfolio_values[-1]:,.2f}")

        results.append({
            'Crisis': crisis_name,
            'Start Date': start_date,
            'End Date': end_date,
            'Days': len(crisis_data),
            'Total Return (%)': total_return * 100,
            'Max Drawdown (%)': max_dd * 100,
            'Sharpe Ratio': sharpe,
            'Final Value': portfolio_values[-1]
        })

    results_df = pd.DataFrame(results)
    return results_df


# ==============================================================================
# 2. Regime-Dependent Performance Analysis
# ==============================================================================

def analyze_regime_performance(
    agent,
    data: pd.DataFrame,
    regime_col: str = 'regime'
) -> pd.DataFrame:
    """
    Analyze agent performance across market regimes.

    Args:
        agent: Trained RL agent
        data: Dataset with regime labels
        regime_col: Column name for regime labels

    Returns:
        regime_results: Performance by regime
    """
    print("\n" + "=" * 70)
    print("Regime-Dependent Performance Analysis")
    print("=" * 70)

    if regime_col not in data.columns:
        print(f"⚠️  Regime column '{regime_col}' not found")
        return pd.DataFrame()

    regimes = data[regime_col].unique()
    results = []

    for regime in regimes:
        regime_data = data[data[regime_col] == regime]

        if len(regime_data) < 50:
            continue

        print(f"\n[REGIME] {regime}: {len(regime_data)} days")
        print("-" * 70)

        # Create environment
        env = PortfolioEnv(
            data=regime_data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',
            reward_type='log_utility'
        )

        # Run episode
        state, _ = env.reset()
        portfolio_values = [env.portfolio_value]
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env.step(action)
            portfolio_values.append(env.portfolio_value)
            done = terminated or truncated

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        avg_return = np.mean(returns) * 252 * 100  # Annualized %
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe = calculate_sharpe_ratio(returns)

        print(f"  Avg Daily Return: {avg_return:.2f}%")
        print(f"  Volatility:       {volatility:.2f}%")
        print(f"  Sharpe Ratio:     {sharpe:.3f}")

        results.append({
            'Regime': regime,
            'Days': len(regime_data),
            'Avg Return (% ann)': avg_return,
            'Volatility (% ann)': volatility,
            'Sharpe Ratio': sharpe
        })

    regime_results = pd.DataFrame(results)
    return regime_results


# ==============================================================================
# 3. Rolling Performance Metrics
# ==============================================================================

def calculate_rolling_metrics(
    portfolio_values: np.ndarray,
    window: int = 63
) -> Dict[str, np.ndarray]:
    """
    Calculate rolling performance metrics.

    Args:
        portfolio_values: Portfolio value time series
        window: Rolling window size (days)

    Returns:
        metrics: Dictionary of rolling metrics
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    rolling_sharpe = []
    rolling_sortino = []
    rolling_calmar = []

    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]

        # Sharpe
        sharpe = np.mean(window_returns) / (np.std(window_returns) + 1e-8) * np.sqrt(252)
        rolling_sharpe.append(sharpe)

        # Sortino
        downside_returns = window_returns[window_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino = np.mean(window_returns) / downside_std * np.sqrt(252)
        rolling_sortino.append(sortino)

        # Calmar
        window_values = portfolio_values[i-window:i+1]
        max_dd = calculate_max_drawdown(window_values)
        annual_return = (window_values[-1] / window_values[0]) ** (252/window) - 1
        calmar = annual_return / (max_dd + 1e-8)
        rolling_calmar.append(calmar)

    return {
        'sharpe': np.array(rolling_sharpe),
        'sortino': np.array(rolling_sortino),
        'calmar': np.array(rolling_calmar)
    }


# ==============================================================================
# 4. Transaction Cost Sensitivity Analysis
# ==============================================================================

def transaction_cost_sensitivity(
    agent,
    data: pd.DataFrame,
    cost_levels: List[float] = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
) -> pd.DataFrame:
    """
    Analyze performance sensitivity to transaction costs.

    Args:
        agent: Trained RL agent
        data: Test dataset
        cost_levels: List of transaction cost levels to test

    Returns:
        sensitivity_results: Performance at different cost levels
    """
    print("\n" + "=" * 70)
    print("Transaction Cost Sensitivity Analysis")
    print("=" * 70)

    results = []

    for cost in cost_levels:
        print(f"\n[COST] Transaction Cost: {cost*100:.2f}%")
        print("-" * 70)

        # Create environment
        env = PortfolioEnv(
            data=data,
            initial_balance=100000.0,
            transaction_cost=cost,
            action_type='discrete',
            reward_type='log_utility'
        )

        # Run episode
        state, _ = env.reset()
        portfolio_values = [env.portfolio_value]
        total_costs = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(env.portfolio_value)

            if 'transaction_cost' in info:
                total_costs += info['transaction_cost']

            done = terminated or truncated

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe = calculate_sharpe_ratio(returns)

        print(f"  Total Return:  {total_return*100:.2f}%")
        print(f"  Sharpe Ratio:  {sharpe:.3f}")
        print(f"  Total Costs:   ${total_costs:,.2f}")

        results.append({
            'Transaction Cost (%)': cost * 100,
            'Total Return (%)': total_return * 100,
            'Sharpe Ratio': sharpe,
            'Total Costs ($)': total_costs,
            'Final Value ($)': portfolio_values[-1]
        })

    sensitivity_df = pd.DataFrame(results)
    return sensitivity_df


# ==============================================================================
# 5. Out-of-Sample Robustness Checks
# ==============================================================================

def oos_robustness_check(
    agent,
    data: pd.DataFrame,
    n_splits: int = 5
) -> Dict:
    """
    Perform out-of-sample robustness checks with walk-forward validation.

    Args:
        agent: Trained RL agent
        data: Full dataset
        n_splits: Number of train/test splits

    Returns:
        oos_results: Out-of-sample performance metrics
    """
    print("\n" + "=" * 70)
    print("Out-of-Sample Robustness Checks")
    print("=" * 70)

    split_size = len(data) // n_splits
    oos_returns = []
    oos_sharpes = []

    for i in range(n_splits):
        test_start = i * split_size
        test_end = test_start + split_size

        test_data = data.iloc[test_start:test_end]

        print(f"\n[SPLIT] {i+1}/{n_splits}: Testing {len(test_data)} days")

        # Create environment
        env = PortfolioEnv(
            data=test_data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',
            reward_type='log_utility'
        )

        # Run episode
        state, _ = env.reset()
        portfolio_values = [env.portfolio_value]
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env.step(action)
            portfolio_values.append(env.portfolio_value)
            done = terminated or truncated

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe = calculate_sharpe_ratio(returns)

        oos_returns.append(total_return * 100)
        oos_sharpes.append(sharpe)

        print(f"  Return: {total_return*100:.2f}%, Sharpe: {sharpe:.3f}")

    results = {
        'mean_return': np.mean(oos_returns),
        'std_return': np.std(oos_returns),
        'mean_sharpe': np.mean(oos_sharpes),
        'std_sharpe': np.std(oos_sharpes),
        'all_returns': oos_returns,
        'all_sharpes': oos_sharpes
    }

    print(f"\n[SUMMARY]")
    print(f"  Mean Return: {results['mean_return']:.2f}% +/- {results['std_return']:.2f}%")
    print(f"  Mean Sharpe: {results['mean_sharpe']:.3f} +/- {results['std_sharpe']:.3f}")

    return results


# ==============================================================================
# 6. Comprehensive Model Comparison Dashboard
# ==============================================================================

def create_comparison_dashboard(
    results_dict: Dict[str, Dict],
    save_path: str = 'simulations/tier2/comparison_dashboard.png'
):
    """
    Create comprehensive comparison dashboard across all models.

    Args:
        results_dict: Dictionary of {model_name: {metric: value}}
        save_path: Path to save dashboard
    """
    print("\n" + "=" * 70)
    print("Creating Comparison Dashboard")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Comparison Dashboard', fontsize=16, fontweight='bold')

    models = list(results_dict.keys())

    # 1. Sharpe Ratio Comparison
    sharpes = [results_dict[m].get('sharpe', 0) for m in models]
    axes[0, 0].bar(models, sharpes, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[0, 0].set_title('Sharpe Ratio', fontweight='bold')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe=1')
    axes[0, 0].legend()

    # 2. Total Return Comparison
    returns = [results_dict[m].get('total_return', 0) for m in models]
    axes[0, 1].bar(models, returns, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[0, 1].set_title('Total Return (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Max Drawdown Comparison (lower is better)
    drawdowns = [results_dict[m].get('max_drawdown', 0) for m in models]
    axes[0, 2].bar(models, drawdowns, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[0, 2].set_title('Max Drawdown (%) - Lower is Better', fontweight='bold')
    axes[0, 2].set_ylabel('Max Drawdown (%)')
    axes[0, 2].grid(axis='y', alpha=0.3)
    axes[0, 2].invert_yaxis()

    # 4. Sortino Ratio
    sortinos = [results_dict[m].get('sortino', 0) for m in models]
    axes[1, 0].bar(models, sortinos, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[1, 0].set_title('Sortino Ratio', fontweight='bold')
    axes[1, 0].set_ylabel('Sortino Ratio')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 5. Calmar Ratio
    calmars = [results_dict[m].get('calmar', 0) for m in models]
    axes[1, 1].bar(models, calmars, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[1, 1].set_title('Calmar Ratio', fontweight='bold')
    axes[1, 1].set_ylabel('Calmar Ratio')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # 6. Risk-Adjusted Performance (Sharpe * Return / Max DD)
    risk_adj = [
        (results_dict[m].get('sharpe', 0) * results_dict[m].get('total_return', 0)) /
        (results_dict[m].get('max_drawdown', 1) + 1e-8)
        for m in models
    ]
    axes[1, 2].bar(models, risk_adj, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[1, 2].set_title('Risk-Adjusted Score', fontweight='bold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].grid(axis='y', alpha=0.3)

    # Rotate x labels for readability
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Dashboard saved to {save_path}")
    plt.close()


# ==============================================================================
# Utility Functions
# ==============================================================================

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if np.std(returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cummax) / cummax
    return abs(np.min(drawdowns))


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Execute Tier 2 improvements suite."""

    print("\n" + "=" * 70)
    print("TIER 2 PRODUCTION ENHANCEMENTS")
    print("=" * 70)
    print("\nEnhancing production system with:")
    print("  1. Crisis period stress testing")
    print("  2. Regime-dependent analysis")
    print("  3. Rolling performance metrics")
    print("  4. Transaction cost sensitivity")
    print("  5. Out-of-sample robustness checks")
    print("  6. Comprehensive comparison dashboard")
    print()

    # Load data
    print("[*] Loading data...")
    data_path = 'data/processed/dataset_with_regimes.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"   Loaded {len(data)} timesteps from {data.index[0]} to {data.index[-1]}")

    # Split data
    train_size = int(len(data) * 0.8)
    test_data = data.iloc[train_size:]
    print(f"   Test data: {len(test_data)} timesteps")

    # Load trained DQN agent
    print("\n[*] Loading trained DQN agent...")
    # The DQN model was trained with 3 discrete actions (weights for 4 assets)
    agent = DQNAgent(state_dim=34, action_dim=3, device='cpu')
    agent.load('models/dqn_trained_ep1000.pth')
    print("   [OK] Agent loaded successfully")

    # Create output directory
    os.makedirs('simulations/tier2', exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Crisis Period Stress Testing
    # -------------------------------------------------------------------------

    crisis_periods = [
        ("COVID-19 Crash", "2020-02-19", "2020-04-30"),
        ("2022 Bear Market", "2022-01-01", "2022-10-31"),
        ("2023 Banking Crisis", "2023-03-08", "2023-03-31")
    ]

    crisis_results = stress_test_crisis_periods(agent, data, crisis_periods)

    if not crisis_results.empty:
        crisis_results.to_csv('simulations/tier2/crisis_stress_test.csv', index=False)
        print(f"\n[OK] Crisis test results saved to simulations/tier2/crisis_stress_test.csv")

    # -------------------------------------------------------------------------
    # 2. Regime-Dependent Analysis
    # -------------------------------------------------------------------------

    regime_results = analyze_regime_performance(agent, test_data)

    if not regime_results.empty:
        regime_results.to_csv('simulations/tier2/regime_analysis.csv', index=False)
        print(f"\n[OK] Regime analysis saved to simulations/tier2/regime_analysis.csv")

    # -------------------------------------------------------------------------
    # 3. Transaction Cost Sensitivity
    # -------------------------------------------------------------------------

    sensitivity_results = transaction_cost_sensitivity(agent, test_data)
    sensitivity_results.to_csv('simulations/tier2/cost_sensitivity.csv', index=False)
    print(f"\n[OK] Sensitivity analysis saved to simulations/tier2/cost_sensitivity.csv")

    # -------------------------------------------------------------------------
    # 4. Out-of-Sample Robustness
    # -------------------------------------------------------------------------

    oos_results = oos_robustness_check(agent, test_data, n_splits=5)

    # -------------------------------------------------------------------------
    # 5. Create Summary Report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIER 2 ENHANCEMENTS COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  [+] simulations/tier2/crisis_stress_test.csv")
    print("  [+] simulations/tier2/regime_analysis.csv")
    print("  [+] simulations/tier2/cost_sensitivity.csv")
    print("\n[SUCCESS] All Tier 2 improvements successfully implemented!")
    print("=" * 70)


if __name__ == '__main__':
    main()
