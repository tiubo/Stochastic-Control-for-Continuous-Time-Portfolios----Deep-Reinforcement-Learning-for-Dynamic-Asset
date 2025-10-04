"""
MLflow Experiment Tracking Integration

Logs training runs, hyperparameters, metrics, and artifacts to MLflow.
Enables experiment comparison, model versioning, and reproducibility.

Usage:
    # Log RL training
    python scripts/mlflow_tracking.py --agent dqn --log-training

    # Log baseline comparison
    python scripts/mlflow_tracking.py --log-baselines

    # Start MLflow UI
    mlflow ui --backend-store-uri file:./mlruns

Then open http://localhost:5000 in your browser.
"""

import mlflow
import mlflow.pytorch
import pandas as pd
import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_mlflow(experiment_name: str = "portfolio-rl"):
    """
    Setup MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    print(f"✅ MLflow experiment: {experiment_name}")
    print(f"📁 Tracking URI: file:./mlruns")


def log_baseline_results():
    """Log baseline strategy results to MLflow."""
    print("\n" + "="*80)
    print("LOGGING BASELINE STRATEGIES TO MLFLOW")
    print("="*80)

    # Load baseline comparison
    comparison_path = Path("simulations/baseline_results/baseline_comparison.csv")
    if not comparison_path.exists():
        print(f"❌ Baseline results not found: {comparison_path}")
        print("   Run: python scripts/test_baseline_strategies.py")
        return

    comparison_df = pd.read_csv(comparison_path)

    for i, row in comparison_df.iterrows():
        strategy = row['Strategy']

        with mlflow.start_run(run_name=f"baseline_{strategy.lower().replace(' ', '_')}"):
            # Log parameters
            mlflow.log_param("strategy_type", "baseline")
            mlflow.log_param("strategy_name", strategy)
            mlflow.log_param("initial_capital", 100000)
            mlflow.log_param("transaction_cost", 0.001)

            # Log metrics
            mlflow.log_metric("total_return_pct", row['Total Return (%)'])
            mlflow.log_metric("annual_return_pct", row['Annual Return (%)'])
            mlflow.log_metric("volatility_pct", row['Volatility (%)'])
            mlflow.log_metric("sharpe_ratio", row['Sharpe Ratio'])
            mlflow.log_metric("sortino_ratio", row['Sortino Ratio'])
            mlflow.log_metric("calmar_ratio", row['Calmar Ratio'])
            mlflow.log_metric("max_drawdown_pct", row['Max Drawdown (%)'])
            mlflow.log_metric("win_rate_pct", row['Win Rate (%)'])
            mlflow.log_metric("avg_turnover_pct", row['Avg Turnover (%)'])

            # Log artifacts
            portfolio_path = Path(f"simulations/baseline_results/{strategy.lower().replace(' ', '_').replace('-', '_')}/portfolio_values.csv")
            if portfolio_path.exists():
                mlflow.log_artifact(str(portfolio_path))

            # Add tags
            mlflow.set_tag("model_type", "classical")
            mlflow.set_tag("date_range", "2014-2024")

            print(f"  ✅ Logged: {strategy}")

    print(f"\n✅ Logged {len(comparison_df)} baseline strategies to MLflow")


def log_rl_training(agent_type: str, model_path: str = None, log_path: str = None):
    """
    Log RL agent training to MLflow.

    Args:
        agent_type: 'dqn', 'ppo', or 'sac'
        model_path: Path to trained model
        log_path: Path to training log
    """
    print(f"\n🤖 Logging {agent_type.upper()} training to MLflow...")

    with mlflow.start_run(run_name=f"rl_{agent_type}"):
        # Log parameters
        mlflow.log_param("agent_type", agent_type)
        mlflow.log_param("model_type", "reinforcement_learning")

        if agent_type == 'dqn':
            mlflow.log_param("action_space", "discrete")
            mlflow.log_param("num_actions", 3)
            mlflow.log_param("episodes", 1000)
            mlflow.log_param("learning_rate", 1e-4)
            mlflow.log_param("gamma", 0.99)
            mlflow.log_param("epsilon_start", 1.0)
            mlflow.log_param("epsilon_end", 0.01)
            mlflow.log_param("memory_size", 100000)

        elif agent_type == 'ppo':
            mlflow.log_param("action_space", "continuous")
            mlflow.log_param("num_envs", 8)
            mlflow.log_param("total_timesteps", 500000)
            mlflow.log_param("learning_rate", 3e-4)
            mlflow.log_param("gamma", 0.99)
            mlflow.log_param("clip_range", 0.2)
            mlflow.log_param("n_steps", 2048)

        elif agent_type == 'sac':
            mlflow.log_param("action_space", "continuous")
            mlflow.log_param("total_timesteps", 500000)
            mlflow.log_param("learning_rate", 3e-4)
            mlflow.log_param("gamma", 0.99)
            mlflow.log_param("tau", 0.005)
            mlflow.log_param("alpha", 0.2)

        # Common parameters
        mlflow.log_param("state_dim", 34)
        mlflow.log_param("device", "cpu")
        mlflow.log_param("data_period", "2014-2024")

        # Log model if exists
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(str(model_path))
            print(f"  ✅ Logged model: {model_path}")

        # Log training log if exists
        if log_path and Path(log_path).exists():
            mlflow.log_artifact(str(log_path))
            print(f"  ✅ Logged training log: {log_path}")

        # Add tags
        mlflow.set_tag("algorithm", agent_type.upper())
        mlflow.set_tag("environment", "PortfolioEnv")

        print(f"✅ Logged {agent_type.upper()} training to MLflow")


def log_comprehensive_comparison():
    """Log comprehensive RL vs baseline comparison."""
    print("\n" + "="*80)
    print("LOGGING COMPREHENSIVE COMPARISON TO MLFLOW")
    print("="*80)

    # Load comprehensive comparison
    comparison_path = Path("simulations/comprehensive_results/comprehensive_comparison.csv")
    if not comparison_path.exists():
        print(f"❌ Comprehensive results not found: {comparison_path}")
        print("   Run: python scripts/run_comprehensive_comparison.py")
        return

    comparison_df = pd.read_csv(comparison_path)

    # Create parent run for comparison
    with mlflow.start_run(run_name="comprehensive_comparison"):
        mlflow.log_param("num_strategies", len(comparison_df))
        mlflow.log_param("comparison_type", "rl_vs_baselines")
        mlflow.log_param("data_period", "2014-2024")

        # Log comparison table
        mlflow.log_artifact(str(comparison_path))

        # Log visualization
        viz_path = Path("simulations/comprehensive_results/comprehensive_comparison.png")
        if viz_path.exists():
            mlflow.log_artifact(str(viz_path))

        # Log summary metrics
        best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        best_return = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
        best_dd = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]

        mlflow.log_metric("best_sharpe_ratio", best_sharpe['Sharpe Ratio'])
        mlflow.log_param("best_sharpe_strategy", best_sharpe['Strategy'])

        mlflow.log_metric("best_total_return", best_return['Total Return (%)'])
        mlflow.log_param("best_return_strategy", best_return['Strategy'])

        mlflow.log_metric("best_max_drawdown", best_dd['Max Drawdown (%)'])
        mlflow.log_param("best_drawdown_strategy", best_dd['Strategy'])

        print(f"\n✅ Logged comprehensive comparison to MLflow")
        print(f"   - Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.3f})")
        print(f"   - Best Return: {best_return['Strategy']} ({best_return['Total Return (%)']:.2f}%)")
        print(f"   - Best Drawdown: {best_dd['Strategy']} ({best_dd['Max Drawdown (%)']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='MLflow experiment tracking')
    parser.add_argument('--log-baselines', action='store_true',
                       help='Log baseline strategy results')
    parser.add_argument('--log-rl', type=str, choices=['dqn', 'ppo', 'sac'],
                       help='Log RL agent training (dqn/ppo/sac)')
    parser.add_argument('--log-comparison', action='store_true',
                       help='Log comprehensive comparison')
    parser.add_argument('--log-all', action='store_true',
                       help='Log everything (baselines + RL + comparison)')
    parser.add_argument('--experiment-name', type=str, default='portfolio-rl',
                       help='MLflow experiment name')

    args = parser.parse_args()

    # Setup MLflow
    setup_mlflow(args.experiment_name)

    # Log based on arguments
    if args.log_all:
        log_baseline_results()
        log_rl_training('dqn', 'models/dqn_trained.pth', 'logs/dqn_training.log')
        log_rl_training('ppo', 'models/ppo/ppo_final.pth', 'logs/ppo_training.log')
        log_rl_training('sac', 'models/sac_trained.pth', 'logs/sac_training.log')
        log_comprehensive_comparison()

    elif args.log_baselines:
        log_baseline_results()

    elif args.log_rl:
        model_paths = {
            'dqn': ('models/dqn_trained.pth', 'logs/dqn_training.log'),
            'ppo': ('models/ppo/ppo_final.pth', 'logs/ppo_training.log'),
            'sac': ('models/sac_trained.pth', 'logs/sac_training.log')
        }
        model_path, log_path = model_paths[args.log_rl]
        log_rl_training(args.log_rl, model_path, log_path)

    elif args.log_comparison:
        log_comprehensive_comparison()

    else:
        print("No logging option specified. Use --help for options.")
        return

    # Instructions
    print("\n" + "="*80)
    print("MLFLOW UI INSTRUCTIONS")
    print("="*80)
    print("\n1. Start MLflow UI:")
    print("   mlflow ui --backend-store-uri file:./mlruns")
    print("\n2. Open browser:")
    print("   http://localhost:5000")
    print("\n3. View experiments, compare runs, and analyze metrics")
    print("="*80)


if __name__ == '__main__':
    main()
