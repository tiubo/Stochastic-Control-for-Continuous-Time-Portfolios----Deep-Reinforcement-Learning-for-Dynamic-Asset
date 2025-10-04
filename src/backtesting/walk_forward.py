"""
Walk-Forward Analysis for Portfolio Allocation Strategies

Walk-forward analysis is a robust backtesting method that:
1. Divides data into multiple train/test windows
2. Trains on in-sample data
3. Tests on out-of-sample data
4. Rolls the window forward
5. Aggregates results across all windows

This prevents look-ahead bias and provides more realistic performance estimates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    train_period: int  # Number of days for training
    test_period: int  # Number of days for testing
    anchored: bool = False  # If True, training window expands; if False, it slides
    min_train_size: int = 252  # Minimum training periods
    step_size: Optional[int] = None  # Step size for rolling (default = test_period)


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis on portfolio strategies.
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward analyzer.

        Args:
            config: Walk-forward configuration
        """
        self.config = config
        if self.config.step_size is None:
            self.config.step_size = self.config.test_period

    def generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test windows for walk-forward analysis.

        Args:
            data: Full dataset

        Returns:
            windows: List of (train_data, test_data) tuples
        """
        windows = []
        total_length = len(data)

        if self.config.anchored:
            # Anchored walk-forward (expanding window)
            start_idx = 0
            train_end = self.config.train_period

            while train_end + self.config.test_period <= total_length:
                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[train_end:train_end + self.config.test_period]

                windows.append((train_data, test_data))

                train_end += self.config.step_size
        else:
            # Rolling walk-forward (sliding window)
            for start_idx in range(0, total_length - self.config.train_period - self.config.test_period + 1, self.config.step_size):
                train_end = start_idx + self.config.train_period
                test_end = train_end + self.config.test_period

                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[train_end:test_end]

                if len(train_data) >= self.config.min_train_size:
                    windows.append((train_data, test_data))

        logger.info(f"Generated {len(windows)} walk-forward windows")
        logger.info(f"Train period: {self.config.train_period} days")
        logger.info(f"Test period: {self.config.test_period} days")
        logger.info(f"Anchored: {self.config.anchored}")

        return windows

    def run_analysis(
        self,
        data: pd.DataFrame,
        train_fn: Callable,
        evaluate_fn: Callable,
        strategy_name: str = "Strategy"
    ) -> Dict:
        """
        Run walk-forward analysis.

        Args:
            data: Full dataset
            train_fn: Function that takes train_data and returns trained model
            evaluate_fn: Function that takes (model, test_data) and returns metrics dict
            strategy_name: Name of strategy

        Returns:
            results: Dictionary with aggregated results
        """
        windows = self.generate_windows(data)

        all_returns = []
        all_metrics = []
        window_results = []

        logger.info(f"Starting walk-forward analysis for {strategy_name}")

        for i, (train_data, test_data) in enumerate(tqdm(windows, desc="Walk-Forward")):
            logger.info(f"\nWindow {i + 1}/{len(windows)}")
            logger.info(f"  Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
            logger.info(f"  Test:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")

            # Train model
            try:
                model = train_fn(train_data)
            except Exception as e:
                logger.error(f"Training failed for window {i + 1}: {e}")
                continue

            # Evaluate on test set
            try:
                metrics = evaluate_fn(model, test_data)

                all_returns.extend(metrics.get('returns', []))
                all_metrics.append(metrics)

                window_results.append({
                    'window_id': i + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'metrics': metrics
                })

                logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
                logger.info(f"  Return: {metrics.get('total_return', 0):.2%}")

            except Exception as e:
                logger.error(f"Evaluation failed for window {i + 1}: {e}")
                continue

        # Aggregate results
        results = {
            'strategy_name': strategy_name,
            'n_windows': len(window_results),
            'window_results': window_results,
            'all_returns': pd.Series(all_returns),
            'aggregated_metrics': self._aggregate_metrics(all_metrics)
        }

        logger.info(f"\nWalk-Forward Analysis Complete")
        logger.info(f"  Total windows: {results['n_windows']}")
        logger.info(f"  Avg Sharpe: {results['aggregated_metrics'].get('avg_sharpe', 0):.3f}")
        logger.info(f"  Avg Return: {results['aggregated_metrics'].get('avg_return', 0):.2%}")

        return results

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all windows."""

        if not all_metrics:
            return {}

        aggregated = {}

        # Average metrics
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            if key != 'returns':  # Skip returns array
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    aggregated[f'avg_{key}'] = np.mean(values)
                    aggregated[f'std_{key}'] = np.std(values)
                    aggregated[f'min_{key}'] = np.min(values)
                    aggregated[f'max_{key}'] = np.max(values)

        # Win rate (percentage of windows with positive Sharpe)
        sharpe_values = [m.get('sharpe_ratio', 0) for m in all_metrics]
        aggregated['win_rate'] = np.mean([s > 0 for s in sharpe_values])

        # Consistency (percentage of windows with positive return)
        return_values = [m.get('total_return', 0) for m in all_metrics]
        aggregated['consistency'] = np.mean([r > 0 for r in return_values])

        return aggregated

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot walk-forward analysis results.

        Args:
            results: Results dictionary from run_analysis
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Sharpe Ratio by Window
        sharpe_values = [w['metrics'].get('sharpe_ratio', 0) for w in results['window_results']]
        window_ids = [w['window_id'] for w in results['window_results']]

        axes[0, 0].bar(window_ids, sharpe_values, alpha=0.7, color='steelblue')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=np.mean(sharpe_values), color='g', linestyle='--', alpha=0.7, label='Mean')
        axes[0, 0].set_title('Sharpe Ratio by Window')
        axes[0, 0].set_xlabel('Window')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative Returns
        returns = results['all_returns']
        cumulative = (1 + returns).cumprod()

        axes[0, 1].plot(cumulative.values, linewidth=2, color='darkgreen')
        axes[0, 1].set_title('Cumulative Returns (Out-of-Sample)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Return Distribution
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=returns.mean(), color='g', linestyle='--', alpha=0.7, label='Mean')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Metrics Comparison
        metrics_names = ['sharpe_ratio', 'max_drawdown', 'total_return', 'sortino_ratio']
        available_metrics = []
        values_list = []

        for metric in metrics_names:
            values = [w['metrics'].get(metric, 0) for w in results['window_results']]
            if any(v != 0 for v in values):
                available_metrics.append(metric.replace('_', ' ').title())
                values_list.append(values)

        if values_list:
            bp = axes[1, 1].boxplot(values_list, labels=available_metrics, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[1, 1].set_title('Metrics Distribution Across Windows')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()

    def compare_strategies(self, results_list: List[Dict], save_path: Optional[str] = None):
        """
        Compare multiple strategies using walk-forward results.

        Args:
            results_list: List of results dictionaries
            save_path: Path to save comparison plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Sharpe Ratio Comparison
        strategy_names = [r['strategy_name'] for r in results_list]
        avg_sharpes = [r['aggregated_metrics'].get('avg_sharpe_ratio', 0) for r in results_list]
        std_sharpes = [r['aggregated_metrics'].get('std_sharpe_ratio', 0) for r in results_list]

        x = np.arange(len(strategy_names))
        axes[0].bar(x, avg_sharpes, yerr=std_sharpes, alpha=0.7, capsize=5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategy_names, rotation=45)
        axes[0].set_title('Average Sharpe Ratio Comparison')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 2. Cumulative Returns Comparison
        for result in results_list:
            returns = result['all_returns']
            cumulative = (1 + returns).cumprod()
            axes[1].plot(cumulative.values, linewidth=2, label=result['strategy_name'], alpha=0.8)

        axes[1].set_title('Cumulative Returns Comparison')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        plt.close()


if __name__ == '__main__':
    # Example usage

    # Generate synthetic data
    dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'return': np.random.randn(len(dates)) * 0.01,
        'price': 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
    }, index=dates)

    # Configure walk-forward
    config = WalkForwardConfig(
        train_period=252,  # 1 year
        test_period=63,    # 3 months
        anchored=False,
        step_size=63
    )

    analyzer = WalkForwardAnalyzer(config)

    # Example train/evaluate functions
    def train_fn(train_data):
        # Simple moving average model
        return {'ma_period': 20}

    def evaluate_fn(model, test_data):
        returns = test_data['return'].values
        return {
            'returns': returns,
            'total_return': (1 + returns).prod() - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': -0.1
        }

    # Run analysis
    results = analyzer.run_analysis(data, train_fn, evaluate_fn, "MA Strategy")

    # Plot results
    analyzer.plot_results(results, save_path='walk_forward_results.png')

    print("\nWalk-forward analysis complete!")
