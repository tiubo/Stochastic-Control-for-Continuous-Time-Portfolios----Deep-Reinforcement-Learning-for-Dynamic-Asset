"""
Performance Benchmarking Suite

Comprehensive performance evaluation including:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown, volatility
- Win rate, profit factor
- Crisis period analysis (2008, 2020, 2022)
- Strategy comparison and statistical tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """Calculate portfolio performance metrics."""

    @staticmethod
    def calculate_returns(
        portfolio_values: pd.Series,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """Calculate return metrics."""
        returns = portfolio_values.pct_change().dropna()

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(portfolio_values) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_daily_return': returns.mean(),
            'volatility': returns.std() * np.sqrt(periods_per_year)
        }

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        return (excess_returns.mean() * periods_per_year) / downside_deviation

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        annualized_return = returns.mean() * periods_per_year
        max_dd = PerformanceMetrics.max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annualized_return / max_dd

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (CVaR / Expected Shortfall)."""
        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega ratio."""
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]

        if returns_below.sum() == 0:
            return np.inf if returns_above.sum() > 0 else 0.0

        return returns_above.sum() / returns_below.sum()

    @staticmethod
    def information_ratio(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Information Ratio."""
        active_returns = strategy_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)

        if tracking_error == 0:
            return 0.0

        return (active_returns.mean() * periods_per_year) / tracking_error

    @staticmethod
    def beta(
        strategy_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate beta relative to market."""
        covariance = np.cov(strategy_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 0.0

        return covariance / market_variance

    @staticmethod
    def alpha(
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Jensen's alpha."""
        beta = PerformanceMetrics.beta(strategy_returns, market_returns)
        daily_rf = risk_free_rate / periods_per_year

        strategy_mean = strategy_returns.mean() * periods_per_year
        market_mean = market_returns.mean() * periods_per_year

        return strategy_mean - (daily_rf + beta * (market_mean - daily_rf))


class StrategyComparison:
    """Compare multiple strategies statistically."""

    @staticmethod
    def compute_all_metrics(
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        returns = portfolio_values.pct_change().dropna()

        metrics = {
            # Return metrics
            **PerformanceMetrics.calculate_returns(portfolio_values, periods_per_year),

            # Risk-adjusted metrics
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, periods_per_year),

            # Risk metrics
            'max_drawdown': PerformanceMetrics.max_drawdown(returns),
            'var_95': PerformanceMetrics.value_at_risk(returns, 0.95),
            'cvar_95': PerformanceMetrics.conditional_var(returns, 0.95),

            # Win metrics
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'omega_ratio': PerformanceMetrics.omega_ratio(returns, 0.0)
        }

        # Benchmark comparison metrics
        if benchmark_values is not None:
            benchmark_returns = benchmark_values.pct_change().dropna()
            metrics.update({
                'information_ratio': PerformanceMetrics.information_ratio(
                    returns, benchmark_returns, periods_per_year
                ),
                'beta': PerformanceMetrics.beta(returns, benchmark_returns),
                'alpha': PerformanceMetrics.alpha(
                    returns, benchmark_returns, risk_free_rate, periods_per_year
                )
            })

        return metrics

    @staticmethod
    def compare_strategies(
        strategies: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """Compare multiple strategies."""
        results = {}

        for name, portfolio_values in strategies.items():
            results[name] = StrategyComparison.compute_all_metrics(
                portfolio_values,
                benchmark,
                risk_free_rate
            )

        return pd.DataFrame(results).T

    @staticmethod
    def statistical_tests(
        strategy1_returns: pd.Series,
        strategy2_returns: pd.Series
    ) -> Dict[str, Tuple[float, float]]:
        """Perform statistical tests between two strategies."""
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(strategy1_returns, strategy2_returns)

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pval = stats.wilcoxon(strategy1_returns, strategy2_returns)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(strategy1_returns, strategy2_returns)

        return {
            't_test': (t_stat, t_pval),
            'wilcoxon': (w_stat, w_pval),
            'ks_test': (ks_stat, ks_pval)
        }


class CrisisAnalysis:
    """Analyze performance during crisis periods."""

    CRISIS_PERIODS = {
        '2008_financial_crisis': ('2008-01-01', '2009-03-31'),
        '2020_covid_crash': ('2020-02-01', '2020-04-30'),
        '2022_rate_hikes': ('2022-01-01', '2022-10-31'),
        'dot_com_bubble': ('2000-03-01', '2002-10-31'),
        'european_debt': ('2011-08-01', '2012-06-30')
    }

    @staticmethod
    def extract_crisis_returns(
        portfolio_values: pd.Series,
        crisis_name: str
    ) -> pd.Series:
        """Extract returns during crisis period."""
        if crisis_name not in CrisisAnalysis.CRISIS_PERIODS:
            raise ValueError(f"Unknown crisis: {crisis_name}")

        start, end = CrisisAnalysis.CRISIS_PERIODS[crisis_name]

        # Filter to crisis period
        mask = (portfolio_values.index >= start) & (portfolio_values.index <= end)
        crisis_values = portfolio_values[mask]

        if len(crisis_values) == 0:
            return pd.Series()

        return crisis_values.pct_change().dropna()

    @staticmethod
    def crisis_performance(
        strategies: Dict[str, pd.Series],
        crisis_name: str
    ) -> pd.DataFrame:
        """Analyze strategy performance during crisis."""
        results = {}

        for name, portfolio_values in strategies.items():
            crisis_returns = CrisisAnalysis.extract_crisis_returns(
                portfolio_values, crisis_name
            )

            if len(crisis_returns) == 0:
                results[name] = {
                    'total_return': np.nan,
                    'max_drawdown': np.nan,
                    'sharpe_ratio': np.nan,
                    'win_rate': np.nan
                }
            else:
                crisis_values = (1 + crisis_returns).cumprod() * portfolio_values.iloc[0]

                results[name] = {
                    'total_return': (crisis_values.iloc[-1] / crisis_values.iloc[0]) - 1,
                    'max_drawdown': PerformanceMetrics.max_drawdown(crisis_returns),
                    'sharpe_ratio': PerformanceMetrics.sharpe_ratio(crisis_returns),
                    'win_rate': PerformanceMetrics.win_rate(crisis_returns)
                }

        return pd.DataFrame(results).T

    @staticmethod
    def all_crisis_analysis(
        strategies: Dict[str, pd.Series]
    ) -> Dict[str, pd.DataFrame]:
        """Analyze all crisis periods."""
        crisis_results = {}

        for crisis_name in CrisisAnalysis.CRISIS_PERIODS.keys():
            try:
                crisis_results[crisis_name] = CrisisAnalysis.crisis_performance(
                    strategies, crisis_name
                )
            except Exception as e:
                print(f"Error analyzing {crisis_name}: {e}")
                continue

        return crisis_results


class RollingMetrics:
    """Calculate rolling performance metrics."""

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 252,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / 252)

        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()

        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return sharpe

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Calculate rolling volatility."""
        return returns.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def rolling_drawdown(
        portfolio_values: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        def max_dd(x):
            cum_ret = (1 + x.pct_change()).cumprod()
            running_max = cum_ret.cummax()
            dd = (cum_ret - running_max) / running_max
            return abs(dd.min())

        return portfolio_values.rolling(window).apply(max_dd, raw=False)


def generate_performance_report(
    strategies: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    output_path: str = 'simulations/performance_report.csv'
) -> pd.DataFrame:
    """Generate comprehensive performance report."""
    # Overall performance
    overall_metrics = StrategyComparison.compare_strategies(strategies, benchmark)

    # Crisis analysis
    crisis_results = CrisisAnalysis.all_crisis_analysis(strategies)

    # Save overall metrics
    overall_metrics.to_csv(output_path)

    # Save crisis metrics
    for crisis_name, crisis_df in crisis_results.items():
        crisis_path = output_path.replace('.csv', f'_crisis_{crisis_name}.csv')
        crisis_df.to_csv(crisis_path)

    print(f"\n{'='*80}")
    print(f"PERFORMANCE REPORT")
    print(f"{'='*80}\n")
    print(overall_metrics.round(4))
    print(f"\n{'='*80}\n")

    return overall_metrics


if __name__ == "__main__":
    # Example usage
    import os

    # Create dummy data for testing
    dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    strategies = {
        'DQN': pd.Series(
            100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001),
            index=dates
        ),
        'PPO': pd.Series(
            100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001),
            index=dates
        ),
        'Merton': pd.Series(
            100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.0008),
            index=dates
        )
    }

    benchmark = pd.Series(
        100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.0007),
        index=dates
    )

    # Generate report
    os.makedirs('simulations', exist_ok=True)
    report = generate_performance_report(strategies, benchmark)
