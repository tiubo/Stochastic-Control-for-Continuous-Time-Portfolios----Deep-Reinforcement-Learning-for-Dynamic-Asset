"""
System Validation Script

Validates all components of the Deep RL Portfolio Allocation system:
- Data pipeline
- Regime detection models
- RL environment
- Agents (DQN, Prioritized DQN, PPO)
- Baseline strategies
- Visualization
- Dashboard components
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def validate_data():
    """Validate data pipeline."""
    print_section("1. DATA PIPELINE VALIDATION")

    # Check raw data
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("[FAIL] Data directory not found!")
        return False

    # Load dataset
    dataset_path = data_dir / "dataset_with_regimes.csv"
    if not dataset_path.exists():
        print(f"[FAIL] Dataset not found: {dataset_path}")
        return False

    data = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

    print(f"[OK] Dataset loaded successfully")
    print(f"   Shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Columns: {', '.join(data.columns[:5])}...")
    print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Check for missing values
    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"[WARN]  Warning: {missing} missing values found")
    else:
        print(f"[OK] No missing values")

    return True

def validate_models():
    """Validate regime detection models."""
    print_section("2. REGIME DETECTION MODELS")

    models_dir = Path("models")

    # Check GMM model
    gmm_path = models_dir / "gmm_regime_detector.pkl"
    if gmm_path.exists():
        print(f"[OK] GMM model found: {gmm_path}")
        print(f"   Size: {gmm_path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[FAIL] GMM model not found")
        return False

    # Check HMM model
    hmm_path = models_dir / "hmm_regime_detector.pkl"
    if hmm_path.exists():
        print(f"[OK] HMM model found: {hmm_path}")
        print(f"   Size: {hmm_path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[FAIL] HMM model not found")
        return False

    return True

def validate_environment():
    """Validate RL environment."""
    print_section("3. RL ENVIRONMENT VALIDATION")

    try:
        from src.environments.portfolio_env import PortfolioEnv

        # Load data
        data = pd.read_csv("data/processed/dataset_with_regimes.csv",
                          index_col=0, parse_dates=True)

        # Create environment
        env = PortfolioEnv(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',
            reward_type='log_utility'
        )

        print(f"[OK] Environment created successfully")
        print(f"   State dim: {env.observation_space.shape[0]}")
        print(f"   Action dim: {env.action_space.n}")
        print(f"   Initial balance: ${env.initial_balance:,.0f}")
        print(f"   Transaction cost: {env.transaction_cost*100:.2f}%")

        # Test reset
        state, _ = env.reset()
        print(f"[OK] Environment reset successful")
        print(f"   State shape: {state.shape}")

        # Test step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"[OK] Environment step successful")
        print(f"   Reward: {reward:.6f}")

        env.close()
        return True

    except Exception as e:
        print(f"[FAIL] Environment validation failed: {e}")
        return False

def validate_agents():
    """Validate RL agents."""
    print_section("4. RL AGENTS VALIDATION")

    try:
        # Validate DQN
        from src.agents.dqn_agent import DQNAgent

        agent_dqn = DQNAgent(
            state_dim=34,
            action_dim=3,
            learning_rate=1e-4
        )
        print(f"[OK] DQN agent initialized")
        print(f"   State dim: 34, Action dim: 3")
        print(f"   Device: {agent_dqn.device}")

        # Validate Prioritized DQN
        from src.agents.prioritized_dqn_agent import PrioritizedDQNAgent

        agent_pdqn = PrioritizedDQNAgent(
            state_dim=34,
            action_dim=3,
            learning_rate=1e-4,
            use_double_dqn=True,
            use_noisy=True
        )
        print(f"[OK] Prioritized DQN agent initialized")
        print(f"   Double DQN: True, Noisy: True")

        # Validate PPO
        from src.agents.ppo_agent import PPOAgent

        agent_ppo = PPOAgent(
            state_dim=34,
            action_dim=3,
            learning_rate=3e-4
        )
        print(f"[OK] PPO agent initialized")
        print(f"   Actor-Critic architecture")

        return True

    except Exception as e:
        print(f"[FAIL] Agent validation failed: {e}")
        return False

def validate_parallel_env():
    """Validate parallel environment wrapper."""
    print_section("5. PARALLEL ENVIRONMENT VALIDATION")

    try:
        from src.environments.parallel_env import make_vec_env, DummyVecEnv
        from src.environments.portfolio_env import PortfolioEnv

        # Load data
        data = pd.read_csv("data/processed/dataset_with_regimes.csv",
                          index_col=0, parse_dates=True)

        def make_env():
            return PortfolioEnv(
                data=data,
                initial_balance=100000.0,
                transaction_cost=0.001,
                action_type='discrete'
            )

        # Create vectorized environment
        vec_env = make_vec_env(
            env_fn=make_env,
            n_envs=4,
            vec_env_cls=DummyVecEnv
        )

        print(f"[OK] Parallel environment created")
        print(f"   Number of environments: {vec_env.num_envs}")
        print(f"   State dim: {vec_env.observation_space.shape[0]}")

        # Test reset
        states = vec_env.reset()
        print(f"[OK] Parallel reset successful")
        print(f"   States shape: {states.shape}")

        # Test step
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"[OK] Parallel step successful")
        print(f"   Rewards: {rewards}")

        vec_env.close()
        return True

    except Exception as e:
        print(f"[FAIL] Parallel environment validation failed: {e}")
        return False

def validate_baselines():
    """Validate baseline strategies."""
    print_section("6. BASELINE STRATEGIES VALIDATION")

    try:
        from src.baselines.merton_strategy import MertonStrategy

        # Load data
        data = pd.read_csv("data/processed/dataset_with_regimes.csv",
                          index_col=0, parse_dates=True)

        # Create Merton strategy
        merton = MertonStrategy(
            risk_free_rate=0.02,
            estimation_window=252,
            rebalance_freq=20
        )

        print(f"[OK] Merton strategy initialized")
        print(f"   Risk-free rate: {merton.risk_free_rate}")
        print(f"   Estimation window: {merton.estimation_window} days")

        return True

    except Exception as e:
        print(f"[FAIL] Baseline validation failed: {e}")
        return False

def validate_benchmarking():
    """Validate performance benchmarking suite."""
    print_section("7. PERFORMANCE BENCHMARKING VALIDATION")

    try:
        from src.backtesting.performance_benchmark import (
            PerformanceMetrics,
            StrategyComparison
        )

        # Create dummy portfolio values
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        portfolio_values = pd.Series(
            100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001),
            index=dates
        )

        # Calculate metrics
        returns = portfolio_values.pct_change().dropna()
        sharpe = PerformanceMetrics.sharpe_ratio(returns)
        max_dd = PerformanceMetrics.max_drawdown(returns)
        sortino = PerformanceMetrics.sortino_ratio(returns)

        print(f"[OK] Performance metrics calculated")
        print(f"   Sharpe ratio: {sharpe:.3f}")
        print(f"   Max drawdown: {max_dd*100:.2f}%")
        print(f"   Sortino ratio: {sortino:.3f}")

        # Test comprehensive metrics
        metrics = StrategyComparison.compute_all_metrics(portfolio_values)
        print(f"[OK] Comprehensive metrics computed")
        print(f"   Total metrics: {len(metrics)}")

        return True

    except Exception as e:
        print(f"[FAIL] Benchmarking validation failed: {e}")
        return False

def validate_dashboard():
    """Validate dashboard components."""
    print_section("8. DASHBOARD COMPONENTS VALIDATION")

    try:
        # Import dashboard modules
        import sys
        sys.path.insert(0, 'app')

        from enhanced_dashboard import DataLoader, MetricsCalculator, RegimeAnalyzer

        # Test data loading
        data = DataLoader.load_dataset("data/processed/dataset_with_regimes.csv")
        if data is not None:
            print(f"[OK] DataLoader working")
            print(f"   Loaded: {data.shape}")

        # Test validation
        is_valid, msg = DataLoader.validate_dataset(data)
        if is_valid:
            print(f"[OK] Dataset validation passed")

        # Test metrics
        if 'price_SPY' in data.columns:
            prices = data['price_SPY'].dropna()
            if len(prices) > 0:
                metrics = MetricsCalculator.calculate_returns(prices)
                print(f"[OK] MetricsCalculator working")
                print(f"   Metrics computed: {len(metrics)}")

        # Test regime analysis
        if 'regime_gmm' in data.columns:
            stats = RegimeAnalyzer.get_regime_stats(data, 'regime_gmm')
            print(f"[OK] RegimeAnalyzer working")
            print(f"   Regimes analyzed: {len(stats)}")

        return True

    except Exception as e:
        print(f"[FAIL] Dashboard validation failed: {e}")
        return False

def validate_dependencies():
    """Validate key dependencies."""
    print_section("9. DEPENDENCIES VALIDATION")

    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit',
        'fastapi': 'FastAPI',
        'optuna': 'Optuna'
    }

    all_valid = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"[OK] {name} installed")
        except ImportError:
            print(f"[FAIL] {name} NOT installed")
            all_valid = False

    return all_valid

def main():
    """Run full system validation."""
    print("\n" + "="*70)
    print("  DEEP RL PORTFOLIO ALLOCATION - SYSTEM VALIDATION")
    print("="*70)

    results = []

    # Run validations
    results.append(("Dependencies", validate_dependencies()))
    results.append(("Data Pipeline", validate_data()))
    results.append(("Regime Models", validate_models()))
    results.append(("RL Environment", validate_environment()))
    results.append(("RL Agents", validate_agents()))
    results.append(("Parallel Environment", validate_parallel_env()))
    results.append(("Baseline Strategies", validate_baselines()))
    results.append(("Performance Benchmarking", validate_benchmarking()))
    results.append(("Dashboard Components", validate_dashboard()))

    # Summary
    print_section("VALIDATION SUMMARY")

    total = len(results)
    passed = sum(1 for _, result in results if result)

    for component, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status:10} - {component}")

    print(f"\n{'='*70}")
    print(f"  Total: {passed}/{total} components validated")

    if passed == total:
        print(f"  Status: [SUCCESS] ALL SYSTEMS OPERATIONAL")
    else:
        print(f"  Status: [WARN]  {total - passed} components need attention")
    print(f"{'='*70}\n")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
