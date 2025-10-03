"""
Unit Tests for Enhanced Streamlit Dashboard
Tests all components with comprehensive coverage
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from enhanced_dashboard import DataLoader, MetricsCalculator, RegimeAnalyzer


class TestDataLoader:
    """Test DataLoader class."""

    def test_validate_dataset_valid(self):
        """Test validation with valid dataset."""
        data = pd.DataFrame({
            'price_SPY': [100, 101, 102],
            'return_SPY': [0.01, 0.01, 0.01],
            'VIX': [15, 16, 17]
        })

        is_valid, msg = DataLoader.validate_dataset(data)
        assert is_valid is True
        assert msg == "Dataset is valid"

    def test_validate_dataset_none(self):
        """Test validation with None."""
        is_valid, msg = DataLoader.validate_dataset(None)
        assert is_valid is False
        assert "None" in msg

    def test_validate_dataset_empty(self):
        """Test validation with empty DataFrame."""
        data = pd.DataFrame()
        is_valid, msg = DataLoader.validate_dataset(data)
        assert is_valid is False
        assert "empty" in msg

    def test_validate_dataset_missing_columns(self):
        """Test validation with missing required columns."""
        data = pd.DataFrame({
            'price_SPY': [100, 101, 102]
        })

        is_valid, msg = DataLoader.validate_dataset(data)
        assert is_valid is False
        assert "Missing" in msg


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)
        return prices

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        return pd.Series(np.random.randn(100) * 0.01)

    def test_calculate_returns_valid(self, sample_prices):
        """Test return calculation with valid data."""
        result = MetricsCalculator.calculate_returns(sample_prices)

        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'annual_return' in result
        assert 'daily_return_mean' in result
        assert 'daily_return_std' in result

        # Check types
        assert isinstance(result['total_return'], float)
        assert isinstance(result['annual_return'], float)

    def test_calculate_sharpe_ratio_valid(self, sample_returns):
        """Test Sharpe ratio calculation with valid data."""
        sharpe = MetricsCalculator.calculate_sharpe_ratio(sample_returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = pd.Series([0.01] * 100)  # Constant returns
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0  # Should return 0 for zero std

    def test_calculate_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty series."""
        returns = pd.Series([])
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_max_drawdown_valid(self, sample_prices):
        """Test max drawdown calculation."""
        drawdown = MetricsCalculator.calculate_max_drawdown(sample_prices)

        assert isinstance(drawdown, float)
        assert 0 <= drawdown <= 1  # Drawdown should be between 0 and 1
        assert not np.isnan(drawdown)

    def test_calculate_max_drawdown_increasing_prices(self):
        """Test max drawdown with continuously increasing prices."""
        prices = pd.Series([100, 105, 110, 115, 120])
        drawdown = MetricsCalculator.calculate_max_drawdown(prices)

        assert drawdown == 0.0  # No drawdown if prices always increase

    def test_calculate_max_drawdown_with_drop(self):
        """Test max drawdown with price drop."""
        prices = pd.Series([100, 110, 90, 95])  # 18.18% drawdown from 110 to 90
        drawdown = MetricsCalculator.calculate_max_drawdown(prices)

        assert drawdown > 0.15  # Should be at least 15%
        assert drawdown < 0.25  # Should be less than 25%


class TestRegimeAnalyzer:
    """Test RegimeAnalyzer class."""

    @pytest.fixture
    def sample_data_with_regime(self):
        """Create sample data with regime column."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'regime_gmm': np.random.choice([0, 1, 2], size=100),
            'regime_hmm': np.random.choice([0, 1, 2], size=100),
            'price_SPY': 100 + np.cumsum(np.random.randn(100))
        }, index=dates)
        return data

    def test_get_regime_stats_gmm(self, sample_data_with_regime):
        """Test regime statistics for GMM."""
        stats = RegimeAnalyzer.get_regime_stats(sample_data_with_regime, 'regime_gmm')

        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty
        assert 'Regime' in stats.columns
        assert 'Count' in stats.columns
        assert 'Percentage' in stats.columns

        # Check percentages sum to 100
        assert abs(stats['Percentage'].sum() - 100.0) < 0.01

    def test_get_regime_stats_hmm(self, sample_data_with_regime):
        """Test regime statistics for HMM."""
        stats = RegimeAnalyzer.get_regime_stats(sample_data_with_regime, 'regime_hmm')

        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty

    def test_get_regime_colors(self):
        """Test regime color mapping."""
        colors = RegimeAnalyzer.get_regime_colors()

        assert isinstance(colors, dict)
        assert 'Bull' in colors
        assert 'Bear' in colors
        assert 'Volatile' in colors

        # Check color values are strings
        for color in colors.values():
            assert isinstance(color, str)


class TestIntegration:
    """Integration tests for combined functionality."""

    @pytest.fixture
    def full_sample_dataset(self):
        """Create comprehensive sample dataset."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        data = pd.DataFrame({
            'price_SPY': 300 + np.cumsum(np.random.randn(100) * 2),
            'price_TLT': 140 + np.cumsum(np.random.randn(100)),
            'price_GLD': 1800 + np.cumsum(np.random.randn(100) * 10),
            'price_BTC-USD': 40000 + np.cumsum(np.random.randn(100) * 500),
            'return_SPY': np.random.randn(100) * 0.01,
            'return_TLT': np.random.randn(100) * 0.005,
            'return_GLD': np.random.randn(100) * 0.008,
            'return_BTC-USD': np.random.randn(100) * 0.03,
            'volatility_SPY': np.abs(np.random.randn(100) * 0.15),
            'volatility_TLT': np.abs(np.random.randn(100) * 0.08),
            'VIX': 15 + np.abs(np.random.randn(100) * 5),
            'Treasury_10Y': 2.5 + np.random.randn(100) * 0.5,
            'regime_gmm': np.random.choice([0, 1, 2], size=100),
            'regime_hmm': np.random.choice([0, 1, 2], size=100)
        }, index=dates)

        return data

    def test_full_workflow_validation(self, full_sample_dataset):
        """Test complete workflow with all components."""
        # Validate data
        is_valid, msg = DataLoader.validate_dataset(full_sample_dataset)
        assert is_valid is True

        # Calculate metrics for each asset
        assets = ['SPY', 'TLT', 'GLD', 'BTC-USD']

        for asset in assets:
            price_col = f'price_{asset}'
            return_col = f'return_{asset}'

            if price_col in full_sample_dataset.columns:
                # Test price metrics
                prices = full_sample_dataset[price_col]
                return_metrics = MetricsCalculator.calculate_returns(prices)
                assert isinstance(return_metrics, dict)

                # Test Sharpe ratio
                if return_col in full_sample_dataset.columns:
                    returns = full_sample_dataset[return_col]
                    sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
                    assert isinstance(sharpe, float)

                # Test drawdown
                drawdown = MetricsCalculator.calculate_max_drawdown(prices)
                assert isinstance(drawdown, float)
                assert 0 <= drawdown <= 1

        # Test regime analysis
        for regime_col in ['regime_gmm', 'regime_hmm']:
            stats = RegimeAnalyzer.get_regime_stats(full_sample_dataset, regime_col)
            assert not stats.empty
            assert len(stats) <= 3  # Should have max 3 regimes

    def test_edge_case_single_observation(self):
        """Test with single observation."""
        data = pd.DataFrame({
            'price_SPY': [100],
            'return_SPY': [0.01],
            'VIX': [15]
        })

        is_valid, msg = DataLoader.validate_dataset(data)
        assert is_valid is True

        # Metrics should handle single observation gracefully
        prices = data['price_SPY']
        returns = data['return_SPY']

        # These should not crash
        return_metrics = MetricsCalculator.calculate_returns(prices)
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
        drawdown = MetricsCalculator.calculate_max_drawdown(prices)

        assert isinstance(return_metrics, dict)
        assert isinstance(sharpe, float)
        assert isinstance(drawdown, float)


def run_all_tests():
    """Run all tests and print summary."""
    import pytest

    print("=" * 80)
    print("RUNNING DASHBOARD UNIT TESTS")
    print("=" * 80)

    # Run pytest
    result = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])

    print("\n" + "=" * 80)
    if result == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    run_all_tests()
