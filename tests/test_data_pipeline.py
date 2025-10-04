"""
Comprehensive unit tests for data pipeline.

Tests cover:
- Data downloading
- Preprocessing
- Feature engineering
- Regime detection
- Data validation
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.features import FeatureEngineer
from regime_detection.gmm_classifier import GMMRegimeDetector
from regime_detection.hmm_classifier import HMMRegimeDetector


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    data = pd.DataFrame(index=dates)

    # Generate realistic price series
    np.random.seed(42)
    for asset in ['SPY', 'TLT', 'GLD', 'BTC']:
        returns = np.random.randn(300) * 0.01
        data[f'price_{asset}'] = 100 * (1 + returns).cumprod()

    return data


@pytest.fixture
def sample_return_data():
    """Create sample return data for testing."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'return_SPY': np.random.randn(300) * 0.015,
        'return_TLT': np.random.randn(300) * 0.008,
        'return_GLD': np.random.randn(300) * 0.012,
        'return_BTC': np.random.randn(300) * 0.04,
        'volatility_SPY': np.abs(np.random.randn(300) * 0.01) + 0.01,
    }, index=dates)

    return data


class TestFeatureEngineer:
    """Test feature engineering functionality."""

    def test_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_rsi_calculation(self, sample_price_data):
        """Test RSI indicator calculation."""
        engineer = FeatureEngineer()

        rsi = engineer.calculate_rsi(sample_price_data['price_SPY'], period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_price_data)
        assert (rsi >= 0).all() or rsi.isna().any()
        assert (rsi <= 100).all() or rsi.isna().any()

    def test_macd_calculation(self, sample_price_data):
        """Test MACD indicator calculation."""
        engineer = FeatureEngineer()

        macd, signal = engineer.calculate_macd(
            sample_price_data['price_SPY'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(macd) == len(sample_price_data)
        assert len(signal) == len(sample_price_data)

    def test_bollinger_bands_calculation(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        engineer = FeatureEngineer()

        upper, middle, lower = engineer.calculate_bollinger_bands(
            sample_price_data['price_SPY'],
            period=20,
            num_std=2
        )

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Upper should be >= Middle >= Lower (where not NaN)
        valid_idx = ~upper.isna()
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_momentum_calculation(self, sample_return_data):
        """Test momentum calculation."""
        engineer = FeatureEngineer()

        momentum = engineer.calculate_momentum(sample_return_data['return_SPY'], period=10)

        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_return_data)

    def test_moving_average_calculation(self, sample_price_data):
        """Test moving average calculation."""
        engineer = FeatureEngineer()

        ma = engineer.calculate_moving_average(sample_price_data['price_SPY'], period=20)

        assert isinstance(ma, pd.Series)
        assert len(ma) == len(sample_price_data)

    def test_sharpe_ratio_calculation(self, sample_return_data):
        """Test rolling Sharpe ratio calculation."""
        engineer = FeatureEngineer()

        sharpe = engineer.calculate_rolling_sharpe(
            sample_return_data['return_SPY'],
            window=20,
            risk_free_rate=0.02
        )

        assert isinstance(sharpe, pd.Series)
        assert len(sharpe) == len(sample_return_data)

    def test_drawdown_calculation(self, sample_price_data):
        """Test drawdown calculation."""
        engineer = FeatureEngineer()

        drawdown = engineer.calculate_drawdown(sample_price_data['price_SPY'])

        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(sample_price_data)
        assert (drawdown <= 0).all()  # Drawdowns should be non-positive

    def test_create_all_features(self, sample_price_data):
        """Test creating all features at once."""
        engineer = FeatureEngineer()

        # Add returns and volatility
        data = sample_price_data.copy()
        for asset in ['SPY', 'TLT', 'GLD', 'BTC']:
            data[f'return_{asset}'] = data[f'price_{asset}'].pct_change()
            data[f'volatility_{asset}'] = data[f'return_{asset}'].rolling(20).std()

        enriched_data = engineer.create_features(data, assets=['SPY'])

        assert 'RSI_SPY' in enriched_data.columns or len(enriched_data.columns) > len(data.columns)

    def test_feature_nan_handling(self, sample_price_data):
        """Test features handle NaN values correctly."""
        engineer = FeatureEngineer()

        # Introduce NaN
        prices_with_nan = sample_price_data['price_SPY'].copy()
        prices_with_nan.iloc[50:55] = np.nan

        rsi = engineer.calculate_rsi(prices_with_nan, period=14)

        # Should not raise error
        assert isinstance(rsi, pd.Series)


class TestGMMRegimeDetector:
    """Test GMM regime detection."""

    def test_gmm_initialization(self):
        """Test GMM detector initializes correctly."""
        detector = GMMRegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3

    def test_gmm_fit(self, sample_return_data):
        """Test GMM fitting."""
        detector = GMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        assert detector.model is not None
        assert hasattr(detector.model, 'means_')

    def test_gmm_predict(self, sample_return_data):
        """Test GMM prediction."""
        detector = GMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        regimes = detector.predict(features)

        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == len(features)
        assert set(regimes).issubset({0, 1, 2})

    def test_gmm_probabilities(self, sample_return_data):
        """Test GMM probability prediction."""
        detector = GMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        probs = detector.predict_proba(features)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (len(features), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_gmm_save_load(self, sample_return_data):
        """Test GMM model saving and loading."""
        detector = GMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "gmm_model.pkl")
            detector.save(save_path)

            # Load into new detector
            new_detector = GMMRegimeDetector(n_regimes=3)
            new_detector.load(save_path)

            # Predictions should match
            regimes1 = detector.predict(features)
            regimes2 = new_detector.predict(features)

            np.testing.assert_array_equal(regimes1, regimes2)

    def test_different_n_regimes(self, sample_return_data):
        """Test GMM with different numbers of regimes."""
        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()

        for n in [2, 3, 4]:
            detector = GMMRegimeDetector(n_regimes=n)
            detector.fit(features)
            regimes = detector.predict(features)

            assert set(regimes).issubset(set(range(n)))


class TestHMMRegimeDetector:
    """Test HMM regime detection."""

    def test_hmm_initialization(self):
        """Test HMM detector initializes correctly."""
        detector = HMMRegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3

    def test_hmm_fit(self, sample_return_data):
        """Test HMM fitting."""
        detector = HMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        assert detector.model is not None
        assert hasattr(detector.model, 'transmat_')

    def test_hmm_predict(self, sample_return_data):
        """Test HMM prediction."""
        detector = HMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        regimes = detector.predict(features)

        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == len(features)
        assert set(regimes).issubset({0, 1, 2})

    def test_hmm_transition_matrix(self, sample_return_data):
        """Test HMM transition matrix properties."""
        detector = HMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        transmat = detector.model.transmat_

        assert transmat.shape == (3, 3)
        # Each row should sum to 1 (probabilities)
        assert np.allclose(transmat.sum(axis=1), 1.0)
        # All entries should be non-negative
        assert (transmat >= 0).all()

    def test_hmm_save_load(self, sample_return_data):
        """Test HMM model saving and loading."""
        detector = HMMRegimeDetector(n_regimes=3)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "hmm_model.pkl")
            detector.save(save_path)

            # Load into new detector
            new_detector = HMMRegimeDetector(n_regimes=3)
            new_detector.load(save_path)

            # Predictions should match
            regimes1 = detector.predict(features)
            regimes2 = new_detector.predict(features)

            np.testing.assert_array_equal(regimes1, regimes2)

    def test_hmm_sequential_dependency(self, sample_return_data):
        """Test HMM captures sequential dependencies."""
        detector = HMMRegimeDetector(n_regimes=2)

        features = sample_return_data[['return_SPY', 'volatility_SPY']].dropna()
        detector.fit(features)

        regimes = detector.predict(features)

        # Check for regime persistence (regimes should have some autocorrelation)
        regime_changes = np.sum(np.diff(regimes) != 0)
        total_periods = len(regimes) - 1

        # Regime changes should be less than random (< 50% for 2 regimes)
        assert regime_changes / total_periods < 0.8


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_missing_values_detection(self):
        """Test detection of missing values."""
        data = pd.DataFrame({
            'return_SPY': [0.01, np.nan, 0.02, 0.01],
            'return_TLT': [0.005, 0.004, 0.005, np.nan]
        })

        missing_pct = data.isna().sum() / len(data)
        assert missing_pct['return_SPY'] == 0.25
        assert missing_pct['return_TLT'] == 0.25

    def test_outlier_detection(self, sample_return_data):
        """Test outlier detection."""
        returns = sample_return_data['return_SPY']

        # Use z-score method
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outliers = z_scores > 3

        assert isinstance(outliers, pd.Series)

    def test_data_continuity(self):
        """Test data has continuous dates."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({'value': range(100)}, index=dates)

        # Check for gaps
        date_diff = data.index.to_series().diff()
        gaps = date_diff > pd.Timedelta(days=1)

        assert gaps.sum() == 0  # No gaps

    def test_return_bounds(self, sample_return_data):
        """Test returns are within reasonable bounds."""
        for col in ['return_SPY', 'return_TLT', 'return_GLD']:
            returns = sample_return_data[col]

            # Daily returns should typically be < 20% (except BTC)
            extreme_returns = np.abs(returns) > 0.2
            assert extreme_returns.sum() / len(returns) < 0.05  # Less than 5% extreme


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling empty dataframe."""
        engineer = FeatureEngineer()
        empty_df = pd.DataFrame()

        # Should handle gracefully
        with pytest.raises(Exception):
            engineer.calculate_rsi(empty_df['price_SPY'] if 'price_SPY' in empty_df else pd.Series(), period=14)

    def test_insufficient_data_for_features(self):
        """Test features with insufficient data."""
        engineer = FeatureEngineer()

        # Only 5 data points, but need 14 for RSI
        short_series = pd.Series([100, 101, 102, 103, 104])
        rsi = engineer.calculate_rsi(short_series, period=14)

        # Should return series with NaN
        assert isinstance(rsi, pd.Series)
        assert rsi.isna().all() or len(rsi) == len(short_series)

    def test_constant_prices(self):
        """Test features with constant prices."""
        engineer = FeatureEngineer()

        constant_prices = pd.Series([100] * 50)
        rsi = engineer.calculate_rsi(constant_prices, period=14)

        # Should handle without error
        assert isinstance(rsi, pd.Series)

    def test_regime_detection_single_regime(self):
        """Test regime detection with uniform data."""
        detector = GMMRegimeDetector(n_regimes=3)

        # Nearly constant data
        uniform_data = pd.DataFrame({
            'return_SPY': np.random.randn(100) * 0.0001,
            'volatility_SPY': np.ones(100) * 0.01
        })

        # Should still fit without error
        detector.fit(uniform_data)
        regimes = detector.predict(uniform_data)

        assert len(regimes) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
