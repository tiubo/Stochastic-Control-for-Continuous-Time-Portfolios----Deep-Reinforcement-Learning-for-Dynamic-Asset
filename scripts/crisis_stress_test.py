"""
Crisis Period Stress Testing Framework

Tests portfolio allocation strategies during major market crises:
- 2008 Financial Crisis (Sep 2008 - Mar 2009)
- COVID-19 Market Crash (Feb 2020 - Apr 2020)
- 2022 Bear Market (Jan 2022 - Oct 2022)

Evaluates agent robustness during extreme volatility and drawdowns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

from src.agents.dqn_agent import DQNAgent
from src.environments.portfolio_env import PortfolioEnv


class CrisisPeriod:
    """Define a crisis period for testing"""

    def __init__(self, name: str, start_date: str, end_date: str, description: str):
        self.name = name
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.description = description

    def __repr__(self):
        return f"CrisisPeriod('{self.name}', {self.start_date.date()} to {self.end_date.date()})"


# Define major crisis periods
CRISIS_PERIODS = [
    CrisisPeriod(
        name="2008 Financial Crisis",
        start_date="2008-09-01",
        end_date="2009-03-31",
        description="Global financial crisis triggered by subprime mortgage collapse"
    ),
    CrisisPeriod(
        name="COVID-19 Crash",
        start_date="2020-02-01",
        end_date="2020-04-30",
        description="Pandemic-driven market crash with record volatility"
    ),
    CrisisPeriod(
        name="2022 Bear Market",
        start_date="2022-01-01",
        end_date="2022-10-31",
        description="Tech-driven bear market with rising interest rates"
    ),
]


class CrisisStressTester:
    """
    Stress test portfolio strategies during crisis periods
    """

    def __init__(self, data_path: str):
        """
        Initialize stress tester

        Args:
            data_path: Path to full dataset
        """
        self.data = pd.read_csv(data_path)

        # Ensure Date column exists
        if 'Date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['Date'])
        elif 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            raise ValueError("No Date/date column found in dataset")

        self.results = {}

    def filter_crisis_data(self, crisis: CrisisPeriod) -> pd.DataFrame:
        """
        Extract data for crisis period

        Args:
            crisis: Crisis period to filter

        Returns:
            Filtered dataframe
        """
        mask = (self.data['date'] >= crisis.start_date) & \
               (self.data['date'] <= crisis.end_date)

        crisis_data = self.data[mask].copy().reset_index(drop=True)

        if len(crisis_data) == 0:
            print(f"WARNING: No data found for {crisis.name}")
            print(f"  Dataset range: {self.data['date'].min()} to {self.data['date'].max()}")
            print(f"  Crisis range: {crisis.start_date} to {crisis.end_date}")

        return crisis_data

    def test_dqn_agent(
        self,
        model_path: str,
        crisis: CrisisPeriod,
        device: str = 'cpu'
    ) -> dict:
        """
        Test DQN agent during crisis period

        Args:
            model_path: Path to trained model
            crisis: Crisis period to test
            device: Device to run on

        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*80}")
        print(f"Testing DQN Agent: {crisis.name}")
        print(f"{'='*80}")

        # Get crisis data
        crisis_data = self.filter_crisis_data(crisis)

        if len(crisis_data) == 0:
            return {'error': 'No data available for crisis period'}

        print(f"Crisis period: {crisis_data['date'].iloc[0]} to {crisis_data['date'].iloc[-1]}")
        print(f"Total days: {len(crisis_data)}")

        # Create environment
        env = PortfolioEnv(crisis_data, window_size=20, action_type='discrete')

        # Load agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNAgent(state_dim, action_dim, device=device)
        agent.load(model_path)

        # Run episode
        state, _ = env.reset()
        done = False

        initial_value = 100000
        portfolio_value = initial_value
        portfolio_values = [portfolio_value]
        returns_list = []
        actions_list = []
        dates_list = [crisis_data['date'].iloc[20]]  # Start after window

        step = 0

        while not done:
            # Get action (greedy)
            action = agent.select_action(state, epsilon=0.0)
            actions_list.append(action)

            # Take step
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Get return
            portfolio_return = info.get('portfolio_return', 0.0)
            returns_list.append(portfolio_return)

            # Update value
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)

            # Track date
            if step + 21 < len(crisis_data):
                dates_list.append(crisis_data['date'].iloc[step + 21])

            state = next_state
            step += 1

        # Calculate metrics
        returns_array = np.array(returns_list)
        portfolio_values_array = np.array(portfolio_values)

        total_return = (portfolio_value / initial_value) - 1
        annualized_vol = returns_array.std() * np.sqrt(252)
        sharpe_ratio = (returns_array.mean() * 252 - 0.02) / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        cumulative = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - cumulative) / cumulative
        max_drawdown = abs(drawdown.min())

        # Worst day
        worst_day_return = returns_array.min()
        best_day_return = returns_array.max()

        results = {
            'crisis_name': crisis.name,
            'start_date': crisis_data['date'].iloc[0],
            'end_date': crisis_data['date'].iloc[-1],
            'days': len(crisis_data),
            'total_return': total_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'worst_day': worst_day_return,
            'best_day': best_day_return,
            'final_value': portfolio_value,
            'portfolio_values': portfolio_values,
            'returns': returns_list,
            'actions': actions_list,
            'dates': dates_list
        }

        # Print summary
        print(f"\n{'='*80}")
        print(f"CRISIS TEST RESULTS: {crisis.name}")
        print(f"{'='*80}")
        print(f"Total Return:         {total_return:>10.2%}")
        print(f"Max Drawdown:         {max_drawdown:>10.2%}")
        print(f"Annualized Vol:       {annualized_vol:>10.2%}")
        print(f"Sharpe Ratio:         {sharpe_ratio:>10.3f}")
        print(f"Worst Day Return:     {worst_day_return:>10.2%}")
        print(f"Best Day Return:      {best_day_return:>10.2%}")
        print(f"Final Portfolio:      ${portfolio_value:>12,.2f}")

        return results

    def test_all_crises(self, model_path: str, device: str = 'cpu') -> pd.DataFrame:
        """
        Test agent across all crisis periods

        Args:
            model_path: Path to trained model
            device: Device to run on

        Returns:
            DataFrame with results for all crises
        """
        all_results = []

        for crisis in CRISIS_PERIODS:
            try:
                results = self.test_dqn_agent(model_path, crisis, device)

                if 'error' not in results:
                    # Store full results
                    self.results[crisis.name] = results

                    # Add to summary
                    summary = {
                        'Crisis': crisis.name,
                        'Period': f"{results['start_date'].date()} to {results['end_date'].date()}",
                        'Days': results['days'],
                        'Total Return': results['total_return'],
                        'Max Drawdown': results['max_drawdown'],
                        'Sharpe Ratio': results['sharpe_ratio'],
                        'Ann. Volatility': results['annualized_volatility'],
                        'Worst Day': results['worst_day'],
                        'Final Value': results['final_value']
                    }
                    all_results.append(summary)
            except Exception as e:
                print(f"Error testing {crisis.name}: {e}")
                continue

        # Create summary DataFrame
        if len(all_results) > 0:
            summary_df = pd.DataFrame(all_results)
            return summary_df
        else:
            print("No crisis periods could be tested (data may not overlap)")
            return pd.DataFrame()

    def plot_crisis_results(self, output_dir: str = 'simulations/crisis_tests'):
        """Generate visualizations for crisis tests"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if len(self.results) == 0:
            print("No results to plot")
            return

        # Set style
        sns.set_style('whitegrid')

        # 1. Portfolio values during each crisis
        fig, axes = plt.subplots(len(self.results), 1, figsize=(15, 5 * len(self.results)))

        if len(self.results) == 1:
            axes = [axes]

        for idx, (crisis_name, results) in enumerate(self.results.items()):
            ax = axes[idx]

            dates = results['dates']
            values = results['portfolio_values'][1:]  # Skip initial value

            # Ensure same length
            min_len = min(len(dates), len(values))
            dates = dates[:min_len]
            values = values[:min_len]

            ax.plot(dates, values, linewidth=2, color='#e74c3c', label='DQN Agent')
            ax.axhline(y=100000, color='black', linestyle='--', alpha=0.5, label='Initial Value')

            ax.set_title(f'{crisis_name} - Portfolio Performance',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Portfolio Value ($)', fontsize=11)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Annotate final value
            final_val = values[-1]
            final_date = dates[-1]
            ax.annotate(f'${final_val:,.0f}',
                       xy=(final_date, final_val),
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path / 'crisis_portfolio_values.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[OK] Saved crisis portfolio values to {output_path / 'crisis_portfolio_values.png'}")

        # 2. Comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Crisis Period Performance Comparison', fontsize=16, fontweight='bold')

        crisis_names = list(self.results.keys())

        # Total Returns
        ax1 = axes[0, 0]
        returns = [self.results[c]['total_return'] * 100 for c in crisis_names]
        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
        ax1.bar(range(len(crisis_names)), returns, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(crisis_names)))
        ax1.set_xticklabels(crisis_names, rotation=45, ha='right')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Total Returns', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Max Drawdowns
        ax2 = axes[0, 1]
        drawdowns = [self.results[c]['max_drawdown'] * 100 for c in crisis_names]
        ax2.bar(range(len(crisis_names)), drawdowns, color='#e74c3c', alpha=0.7)
        ax2.set_xticks(range(len(crisis_names)))
        ax2.set_xticklabels(crisis_names, rotation=45, ha='right')
        ax2.set_ylabel('Max Drawdown (%)')
        ax2.set_title('Maximum Drawdowns', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Sharpe Ratios
        ax3 = axes[1, 0]
        sharpes = [self.results[c]['sharpe_ratio'] for c in crisis_names]
        colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]
        ax3.bar(range(len(crisis_names)), sharpes, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(crisis_names)))
        ax3.set_xticklabels(crisis_names, rotation=45, ha='right')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratios', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.grid(True, alpha=0.3)

        # Volatility
        ax4 = axes[1, 1]
        vols = [self.results[c]['annualized_volatility'] * 100 for c in crisis_names]
        ax4.bar(range(len(crisis_names)), vols, color='#3498db', alpha=0.7)
        ax4.set_xticks(range(len(crisis_names)))
        ax4.set_xticklabels(crisis_names, rotation=45, ha='right')
        ax4.set_ylabel('Annualized Volatility (%)')
        ax4.set_title('Volatility Levels', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'crisis_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved crisis comparison to {output_path / 'crisis_comparison.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Crisis Period Stress Testing')
    parser.add_argument('--data', type=str, default='data/processed/dataset_with_regimes.csv',
                       help='Path to dataset')
    parser.add_argument('--model', type=str, default='models/dqn_trained_ep1000.pth',
                       help='Path to trained DQN model')
    parser.add_argument('--output', type=str, default='simulations/crisis_tests',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CRISIS PERIOD STRESS TESTING FRAMEWORK")
    print("="*80)

    # Initialize tester
    tester = CrisisStressTester(args.data)

    # Test all crisis periods
    summary_df = tester.test_all_crises(args.model, args.device)

    if len(summary_df) > 0:
        # Display summary
        print("\n" + "="*80)
        print("CRISIS TEST SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))

        # Save summary
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path / 'crisis_summary.csv', index=False)
        print(f"\n[OK] Saved summary to {output_path / 'crisis_summary.csv'}")

        # Generate plots
        tester.plot_crisis_results(args.output)

        print("\n" + "="*80)
        print("STRESS TESTING COMPLETE!")
        print("="*80)
    else:
        print("\n[WARNING] No crisis periods could be tested")
        print("Dataset may not cover the crisis periods defined")


if __name__ == '__main__':
    main()
