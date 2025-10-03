"""
Generate All Visualizations
Creates plots for EDA, regime analysis, and performance comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.visualization.plots import PortfolioVisualizer
from src.regime_detection.gmm_classifier import GMMRegimeDetector
from src.regime_detection.hmm_classifier import HMMRegimeDetector

def main():
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    try:
        data = pd.read_csv("data/processed/dataset_with_regimes.csv",
                          index_col=0, parse_dates=True)
        print(f"   Loaded: {data.shape}")
    except FileNotFoundError:
        print("   ERROR: Dataset not found!")
        print("   Please run: python scripts/simple_preprocess.py")
        print("   Then run: python scripts/train_regime_models.py")
        return

    # Initialize visualizer
    viz = PortfolioVisualizer(save_dir="docs/figures")

    # 1. Price trajectories
    print("\n2. Creating price trajectory plots...")
    viz.plot_price_trajectories(data, save_name="asset_prices.png")

    # 2. Correlation matrix
    print("\n3. Creating correlation matrix...")
    viz.plot_correlation_matrix(data, save_name="return_correlation.png")

    # 3. Volatility time series
    print("\n4. Creating volatility time series...")
    viz.plot_volatility_timeseries(data, save_name="volatility_vix.png")

    # 4. Regime-colored prices (GMM)
    print("\n5. Creating regime-colored price plots (GMM)...")
    viz.plot_regime_colored_prices(
        data,
        regime_col='regime_gmm',
        price_col='price_SPY',
        save_name="spy_regime_gmm.png"
    )

    # 5. Regime-colored prices (HMM)
    print("\n6. Creating regime-colored price plots (HMM)...")
    viz.plot_regime_colored_prices(
        data,
        regime_col='regime_hmm',
        price_col='price_SPY',
        save_name="spy_regime_hmm.png"
    )

    # 6. Regime statistics (GMM)
    print("\n7. Creating regime statistics plots...")
    try:
        gmm_detector = GMMRegimeDetector()
        gmm_detector.load("models/gmm_regime_detector.pkl")

        returns = data[[col for col in data.columns if col.startswith('return_')]]
        vix = data['VIX']

        gmm_stats = gmm_detector.get_regime_statistics(returns, vix)
        viz.plot_regime_statistics(gmm_stats, save_name="gmm_regime_stats.png")
    except Exception as e:
        print(f"   Warning: Could not create GMM stats plot: {e}")

    # 7. Create mock comparison plots (will be updated after training)
    print("\n8. Creating placeholder performance plots...")

    # Mock wealth trajectories
    dates = data.index[-500:]
    initial_value = 100000

    # Simulate some trajectories for demonstration
    np.random.seed(42)
    buy_hold = initial_value * np.cumprod(1 + np.random.normal(0.0003, 0.01, len(dates)))
    merton = initial_value * np.cumprod(1 + np.random.normal(0.0004, 0.008, len(dates)))
    dqn_mock = initial_value * np.cumprod(1 + np.random.normal(0.0005, 0.009, len(dates)))

    strategies = {
        'Buy & Hold (60/40)': buy_hold,
        'Merton Strategy': merton,
        'DQN Agent (Placeholder)': dqn_mock
    }

    viz.plot_wealth_comparison(strategies, dates, save_name="wealth_comparison_placeholder.png")
    viz.plot_drawdown_comparison(strategies, dates, save_name="drawdown_placeholder.png")

    # Risk-return scatter
    strategies_metrics = {
        'Buy & Hold': {'return': 0.08, 'volatility': 0.12},
        'Merton': {'return': 0.10, 'volatility': 0.10},
        'DQN (Placeholder)': {'return': 0.12, 'volatility': 0.11}
    }
    viz.plot_risk_return_scatter(strategies_metrics, save_name="risk_return_placeholder.png")

    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated plots:")
    print("\nEDA (Exploratory Data Analysis):")
    print("  - docs/figures/eda/asset_prices.png")
    print("  - docs/figures/eda/return_correlation.png")
    print("  - docs/figures/eda/volatility_vix.png")
    print("\nRegime Analysis:")
    print("  - docs/figures/regimes/spy_regime_gmm.png")
    print("  - docs/figures/regimes/spy_regime_hmm.png")
    print("  - docs/figures/regimes/gmm_regime_stats.png")
    print("\nPerformance (Placeholders - will update after DQN training):")
    print("  - docs/figures/performance/wealth_comparison_placeholder.png")
    print("  - docs/figures/performance/drawdown_placeholder.png")
    print("  - docs/figures/performance/risk_return_placeholder.png")
    print("\n" + "=" * 80)
    print("\nNote: Performance plots are placeholders.")
    print("After DQN training, run scripts/compare_strategies.py for real results.")
    print("=" * 80)

if __name__ == "__main__":
    main()
