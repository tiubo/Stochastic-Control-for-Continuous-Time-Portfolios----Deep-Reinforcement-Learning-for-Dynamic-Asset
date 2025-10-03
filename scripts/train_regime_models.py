"""
Train Regime Detection Models (GMM and HMM)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.regime_detection.gmm_classifier import GMMRegimeDetector
from src.regime_detection.hmm_classifier import HMMRegimeDetector

def main():
    print("=" * 80)
    print("REGIME DETECTION MODEL TRAINING")
    print("=" * 80)

    # Load processed data
    print("\n1. Loading processed dataset...")
    try:
        data = pd.read_csv("data/processed/complete_dataset.csv", index_col=0, parse_dates=True)
        print(f"   Loaded: {data.shape}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    except FileNotFoundError:
        print("   ERROR: Processed dataset not found!")
        print("   Please run: python scripts/simple_preprocess.py first")
        return

    # Extract returns and VIX
    return_cols = [col for col in data.columns if col.startswith('return_')]
    returns = data[return_cols].copy()
    vix = data['VIX'].copy()

    print(f"\n2. Features for regime detection:")
    print(f"   - Returns: {returns.shape}")
    print(f"   - VIX: {vix.shape}")

    # Train GMM
    print("\n" + "=" * 80)
    print("TRAINING GAUSSIAN MIXTURE MODEL (GMM)")
    print("=" * 80)

    gmm_detector = GMMRegimeDetector(n_regimes=3, random_state=42)
    gmm_detector.fit(returns, vix)

    # Predict regimes
    gmm_regimes = gmm_detector.predict(returns, vix)

    # Get statistics
    gmm_stats = gmm_detector.get_regime_statistics(returns, vix)
    print("\nGMM Regime Statistics:")
    print(gmm_stats.to_string(index=False))

    # Save GMM model
    os.makedirs("models", exist_ok=True)
    gmm_detector.save("models/gmm_regime_detector.pkl")

    # Save regime labels
    os.makedirs("data/regime_labels", exist_ok=True)
    gmm_regimes.to_csv("data/regime_labels/gmm_regimes.csv")
    print(f"\nGMM regimes saved to data/regime_labels/gmm_regimes.csv")

    # Train HMM
    print("\n" + "=" * 80)
    print("TRAINING HIDDEN MARKOV MODEL (HMM)")
    print("=" * 80)

    hmm_detector = HMMRegimeDetector(n_regimes=3, n_iter=100, random_state=42)
    hmm_detector.fit(returns)

    # Predict regimes
    hmm_regimes = hmm_detector.predict(returns)

    # Get statistics
    hmm_stats = hmm_detector.get_regime_statistics(returns)
    print("\nHMM Regime Statistics:")
    print(hmm_stats.to_string(index=False))

    # Transition matrix
    print("\nHMM Transition Matrix:")
    transition_matrix = hmm_detector.get_transition_matrix()
    print(transition_matrix.round(3).to_string())

    # Save HMM model
    hmm_detector.save("models/hmm_regime_detector.pkl")

    # Save regime labels
    hmm_regimes.to_csv("data/regime_labels/hmm_regimes.csv")
    print(f"\nHMM regimes saved to data/regime_labels/hmm_regimes.csv")

    # Compare regime assignments
    print("\n" + "=" * 80)
    print("REGIME COMPARISON (GMM vs HMM)")
    print("=" * 80)

    # Align regimes
    common_idx = gmm_regimes.index.intersection(hmm_regimes.index)
    gmm_aligned = gmm_regimes.loc[common_idx]
    hmm_aligned = hmm_regimes.loc[common_idx]

    # Agreement percentage
    agreement = (gmm_aligned == hmm_aligned).sum() / len(common_idx) * 100
    print(f"\nRegime agreement: {agreement:.1f}%")

    # Cross-tabulation
    print("\nCross-tabulation (GMM vs HMM):")
    crosstab = pd.crosstab(gmm_aligned, hmm_aligned, margins=True)
    print(crosstab)

    # Add regimes to dataset
    print("\n" + "=" * 80)
    print("ADDING REGIMES TO DATASET")
    print("=" * 80)

    data_with_regimes = data.copy()
    data_with_regimes['regime_gmm'] = gmm_regimes
    data_with_regimes['regime_hmm'] = hmm_regimes

    # Fill any missing regime values with mode
    data_with_regimes['regime_gmm'].fillna(data_with_regimes['regime_gmm'].mode()[0], inplace=True)
    data_with_regimes['regime_hmm'].fillna(data_with_regimes['regime_hmm'].mode()[0], inplace=True)

    # Save enhanced dataset
    data_with_regimes.to_csv("data/processed/dataset_with_regimes.csv")
    print(f"\nEnhanced dataset saved: {data_with_regimes.shape}")
    print(f"File: data/processed/dataset_with_regimes.csv")

    print("\n" + "=" * 80)
    print("REGIME TRAINING COMPLETE!")
    print("=" * 80)
    print("\nSaved files:")
    print("  - models/gmm_regime_detector.pkl")
    print("  - models/hmm_regime_detector.pkl")
    print("  - data/regime_labels/gmm_regimes.csv")
    print("  - data/regime_labels/hmm_regimes.csv")
    print("  - data/processed/dataset_with_regimes.csv")
    print("\nNext step:")
    print("  python scripts/train_dqn.py --episodes 500")
    print("=" * 80)

if __name__ == "__main__":
    main()
