"""
Hyperparameter Optimization Framework using Optuna

Automated hyperparameter tuning for:
- DQN/PPO agents
- Network architectures
- Learning rates, discount factors
- Exploration strategies
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.agents.prioritized_dqn_agent import PrioritizedDQNAgent
from src.environments.portfolio_env import PortfolioEnv

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Hyperparameter optimization for RL agents."""

    def __init__(
        self,
        agent_type: str = 'ppo',
        n_trials: int = 100,
        n_eval_episodes: int = 10,
        eval_freq: int = 5000,
        max_steps: int = 50000,
        pruner_type: str = 'median',
        sampler_type: str = 'tpe'
    ):
        self.agent_type = agent_type
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.max_steps = max_steps

        # Pruner for early stopping of unpromising trials
        if pruner_type == 'median':
            self.pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=self.eval_freq,
                interval_steps=self.eval_freq
            )
        else:
            self.pruner = optuna.pruners.NopPruner()

        # Sampler for hyperparameter search
        if sampler_type == 'tpe':
            self.sampler = TPESampler(n_startup_trials=10)
        else:
            self.sampler = optuna.samplers.RandomSampler()

    def sample_ppo_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample PPO hyperparameters."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'clip_epsilon': trial.suggest_float('clip_epsilon', 0.1, 0.4),
            'value_coef': trial.suggest_float('value_coef', 0.3, 0.7),
            'entropy_coef': trial.suggest_float('entropy_coef', 0.0, 0.1),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'hidden_dims': trial.suggest_categorical('hidden_dims', [
                [128, 128],
                [256, 256],
                [512, 256],
                [256, 256, 128]
            ])
        }

    def sample_dqn_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample DQN hyperparameters."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'buffer_capacity': trial.suggest_categorical('buffer_capacity', [50000, 100000, 200000]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'target_update_freq': trial.suggest_int('target_update_freq', 500, 2000, step=500),
            'per_alpha': trial.suggest_float('per_alpha', 0.4, 0.8),
            'per_beta_start': trial.suggest_float('per_beta_start', 0.3, 0.6),
            'hidden_dims': trial.suggest_categorical('hidden_dims', [
                [128, 64],
                [256, 128],
                [256, 256],
                [512, 256]
            ]),
            'use_double_dqn': trial.suggest_categorical('use_double_dqn', [True, False]),
            'use_noisy': trial.suggest_categorical('use_noisy', [True, False])
        }

    def evaluate_agent(
        self,
        agent,
        env: PortfolioEnv,
        n_episodes: int = 10
    ) -> float:
        """Evaluate agent performance."""
        episode_rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                if self.agent_type == 'ppo':
                    action, _, _ = agent.select_action(state, deterministic=True)
                    # For PPO, action is continuous weights
                    # Convert to discrete action for discrete env
                    action_idx = np.argmax(action)
                else:
                    action_idx = agent.select_action(state, epsilon=0.0)

                next_state, reward, terminated, truncated, _ = env.step(action_idx)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)

        return np.mean(episode_rewards)

    def objective_ppo(self, trial: optuna.Trial, env_fn: Callable) -> float:
        """Objective function for PPO optimization."""
        # Sample hyperparameters
        params = self.sample_ppo_hyperparameters(trial)

        # Create environment
        env = env_fn()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=params['hidden_dims'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            clip_epsilon=params['clip_epsilon'],
            value_coef=params['value_coef'],
            entropy_coef=params['entropy_coef'],
            max_grad_norm=params['max_grad_norm'],
            n_epochs=params['n_epochs'],
            batch_size=params['batch_size']
        )

        # Rollout buffer
        buffer = RolloutBuffer(
            buffer_size=2048,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # Training loop
        state, _ = env.reset()
        total_steps = 0
        best_eval_reward = -np.inf

        while total_steps < self.max_steps:
            # Collect rollout
            for _ in range(2048):
                action, log_prob, value = agent.select_action(state)

                # Convert continuous action to discrete for env
                action_idx = np.argmax(action)
                next_state, reward, terminated, truncated, _ = env.step(action_idx)
                done = terminated or truncated

                buffer.add(state, action, reward, value, log_prob, done)

                state = next_state if not done else env.reset()[0]
                total_steps += 1

                if total_steps >= self.max_steps:
                    break

            # Update agent
            agent.update(buffer)
            buffer.reset()

            # Evaluate periodically
            if total_steps % self.eval_freq == 0:
                eval_reward = self.evaluate_agent(agent, env, self.n_eval_episodes)
                best_eval_reward = max(best_eval_reward, eval_reward)

                # Report to Optuna
                trial.report(eval_reward, total_steps)

                # Prune if needed
                if trial.should_prune():
                    raise optuna.TrialPruned()

        env.close()
        return best_eval_reward

    def objective_dqn(self, trial: optuna.Trial, env_fn: Callable) -> float:
        """Objective function for DQN optimization."""
        # Sample hyperparameters
        params = self.sample_dqn_hyperparameters(trial)

        # Create environment
        env = env_fn()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create agent
        agent = PrioritizedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=params['hidden_dims'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            buffer_capacity=params['buffer_capacity'],
            batch_size=params['batch_size'],
            target_update_freq=params['target_update_freq'],
            use_double_dqn=params['use_double_dqn'],
            use_noisy=params['use_noisy'],
            per_alpha=params['per_alpha'],
            per_beta_start=params['per_beta_start']
        )

        # Training loop
        state, _ = env.reset()
        total_steps = 0
        episode_reward = 0
        best_eval_reward = -np.inf

        # Exploration schedule
        epsilon_start = 1.0 if not params['use_noisy'] else 0.0
        epsilon_end = 0.01
        epsilon_decay = 0.995

        epsilon = epsilon_start

        while total_steps < self.max_steps:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Calculate TD error for prioritization
            td_error = reward  # Simplified; actual TD error computed in agent

            agent.replay_buffer.add(state, action, reward, next_state, done, td_error)
            episode_reward += reward

            # Update agent
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            state = next_state if not done else env.reset()[0]

            if done:
                episode_reward = 0
                epsilon = max(epsilon_end, epsilon * epsilon_decay)

            total_steps += 1

            # Evaluate periodically
            if total_steps % self.eval_freq == 0:
                eval_reward = self.evaluate_agent(agent, env, self.n_eval_episodes)
                best_eval_reward = max(best_eval_reward, eval_reward)

                # Report to Optuna
                trial.report(eval_reward, total_steps)

                # Prune if needed
                if trial.should_prune():
                    raise optuna.TrialPruned()

        env.close()
        return best_eval_reward

    def optimize(
        self,
        env_fn: Callable,
        study_name: str = "portfolio_optimization",
        storage: Optional[str] = None
    ) -> optuna.Study:
        """Run hyperparameter optimization."""

        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            pruner=self.pruner,
            sampler=self.sampler,
            storage=storage,
            load_if_exists=True
        )

        # Define objective
        if self.agent_type == 'ppo':
            objective = lambda trial: self.objective_ppo(trial, env_fn)
        elif self.agent_type == 'dqn':
            objective = lambda trial: self.objective_dqn(trial, env_fn)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Run optimization
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Print results
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimization complete for {self.agent_type.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*60}\n")

        return study

    def save_study_results(self, study: optuna.Study, filepath: str):
        """Save optimization results to CSV."""
        df = study.trials_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Study results saved to {filepath}")

    def plot_optimization_history(self, study: optuna.Study, filepath: str):
        """Plot optimization history."""
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import plotly.io as pio

            # Optimization history
            fig1 = plot_optimization_history(study)
            pio.write_html(fig1, filepath.replace('.png', '_history.html'))

            # Parameter importances
            fig2 = plot_param_importances(study)
            pio.write_html(fig2, filepath.replace('.png', '_importances.html'))

            logger.info(f"Optimization plots saved to {filepath}")
        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")


def run_hyperparameter_search(
    agent_type: str = 'ppo',
    data_path: str = 'data/processed/dataset_with_regimes.csv',
    n_trials: int = 50,
    max_steps: int = 50000,
    output_dir: str = 'models/hyperparameter_search'
):
    """Run hyperparameter search for agent."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Environment factory
    def make_env():
        return PortfolioEnv(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',
            reward_type='log_utility'
        )

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        agent_type=agent_type,
        n_trials=n_trials,
        n_eval_episodes=10,
        eval_freq=5000,
        max_steps=max_steps
    )

    # Run optimization
    study = optimizer.optimize(
        env_fn=make_env,
        study_name=f"{agent_type}_portfolio_optimization",
        storage=f"sqlite:///{output_dir}/{agent_type}_study.db"
    )

    # Save results
    optimizer.save_study_results(
        study,
        f"{output_dir}/{agent_type}_hyperparameter_results.csv"
    )

    optimizer.plot_optimization_history(
        study,
        f"{output_dir}/{agent_type}_optimization.png"
    )

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter optimization for RL agents')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'dqn'],
                        help='Agent type to optimize')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--max-steps', type=int, default=50000,
                        help='Maximum training steps per trial')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/dataset_with_regimes.csv',
                        help='Path to processed dataset')
    parser.add_argument('--output-dir', type=str,
                        default='models/hyperparameter_search',
                        help='Output directory for results')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run optimization
    study = run_hyperparameter_search(
        agent_type=args.agent,
        data_path=args.data_path,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )
