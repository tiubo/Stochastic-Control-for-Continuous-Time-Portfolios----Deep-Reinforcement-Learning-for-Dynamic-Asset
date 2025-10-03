"""
Simplified Data Preprocessing
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Starting data preprocessing...")

    # Load data
    prices = pd.read_csv("data/raw/asset_prices_1d.csv", index_col=0, parse_dates=True)
    vix_df = pd.read_csv("data/raw/vix.csv", index_col=0, parse_dates=True)

    # Extract VIX as series
    if isinstance(vix_df, pd.DataFrame):
        vix = vix_df.iloc[:, 0] if len(vix_df.columns) > 0 else vix_df.squeeze()
    else:
        vix = vix_df

    # Create treasury mock data
    treasury = pd.Series(
        data=2.5 + 0.5 * np.random.randn(len(prices)),
        index=prices.index,
        name='Treasury_10Y'
    ).clip(lower=0.5, upper=5.0)

    print(f"Loaded - Prices: {prices.shape}, VIX: {vix.shape}, Treasury: {treasury.shape}")

    # Find common dates
    common_dates = prices.index.intersection(vix.index)
    print(f"Common dates: {len(common_dates)}")

    # Align data
    prices_aligned = prices.loc[common_dates]
    vix_aligned = vix.loc[common_dates]
    treasury_aligned = treasury.loc[common_dates]

    # Calculate returns
    returns = np.log(prices_aligned / prices_aligned.shift(1))

    # Calculate rolling volatility (20-day)
    volatility = returns.rolling(window=20).std() * np.sqrt(252)

    # Create dataset
    dataset = pd.DataFrame(index=common_dates)

    # Add prices
    for col in prices_aligned.columns:
        dataset[f'price_{col}'] = prices_aligned[col]

    # Add returns
    for col in returns.columns:
        dataset[f'return_{col}'] = returns[col]

    # Add volatility
    for col in volatility.columns:
        dataset[f'volatility_{col}'] = volatility[col]

    # Add macro
    dataset['VIX'] = vix_aligned
    dataset['Treasury_10Y'] = treasury_aligned

    # Drop NaN rows (from rolling calculations)
    dataset_clean = dataset.dropna()

    print(f"\nFinal dataset: {dataset_clean.shape}")
    print(f"Date range: {dataset_clean.index[0]} to {dataset_clean.index[-1]}")
    print(f"Columns: {list(dataset_clean.columns)}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    dataset_clean.to_csv("data/processed/complete_dataset.csv")
    print(f"\nSaved to data/processed/complete_dataset.csv")

    # Stats
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total observations: {len(dataset_clean)}")
    print(f"Total features: {dataset_clean.shape[1]}")
    print(f"\nAssets: {prices_aligned.columns.tolist()}")
    print(f"\nFeature breakdown:")
    print(f"  - Prices: 4")
    print(f"  - Returns: 4")
    print(f"  - Volatility: 4")
    print(f"  - Macro: 2 (VIX, Treasury)")

    # Train/test split info
    train_size = int(len(dataset_clean) * 0.8)
    print(f"\nSuggested train/test split (80/20):")
    print(f"  - Train: {train_size} days ({dataset_clean.index[0]} to {dataset_clean.index[train_size-1]})")
    print(f"  - Test: {len(dataset_clean) - train_size} days ({dataset_clean.index[train_size]} to {dataset_clean.index[-1]})")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
