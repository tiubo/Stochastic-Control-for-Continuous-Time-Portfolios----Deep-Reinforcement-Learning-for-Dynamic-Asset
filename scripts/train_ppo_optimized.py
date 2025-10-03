"""
Optimized PPO Training Script with Parallel Environments

Features:
- Vectorized environments for faster data collection
- Automatic checkpointing
- TensorBoard logging
- Performance tracking
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import logging
from pathlib import Path

from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.environments.portfolio_env import PortfolioEnv
from src.environments.parallel_env import make_vec_env, DummyVecEnv, VecNormalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_env_fn(data: pd.DataFrame, rank: int = 0):
    """Environment factory function."""
    def _init():
        # Split data for each env (optional: use different seeds)
        env = PortfolioEnv(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            action_type='discrete',  # PPO works with both
            reward_type='log_utility'
        )
        return env
    return _init


def train_ppo_optimized(
    data_path: str = 'data/processed/dataset_with_regimes.csv',
    n_envs: int = 4,
    total_timesteps: int = 500000,
    n_steps: int = 2048,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    save_freq: int = 50000,
    log_freq: int = 2048,
    output_dir: str = 'models/ppo_optimized',
    device: str = 'cpu'
):
    """Train PPO agent with optimizations."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting PPO training: {run_dir}")
    logger.info(f"Using device: {device}")
    logger.info(f"Number of parallel environments: {n_envs}")

    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded dataset: {data.shape}")

    # Create vectorized environment
    vec_env = make_vec_env(
        env_fn=make_env_fn(data),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv  # Use DummyVecEnv for compatibility
    )

    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        obs_norm=True,
        ret_norm=True,
        gamma=gamma
    )

    # Get environment specs
    state_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.n

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device
    )

    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = np.zeros(n_envs)
    current_episode_length = np.zeros(n_envs)

    # Training loop
    states = vec_env.reset()
    total_steps = 0
    n_updates = 0

    logger.info("Starting training loop...")

    try:
        while total_steps < total_timesteps:
            # Collect rollout
            for step in range(n_steps):
                # Select actions for all environments
                actions_batch = []
                log_probs_batch = []
                values_batch = []

                for env_idx in range(n_envs):
                    action, log_prob, value = agent.select_action(states[env_idx])
                    actions_batch.append(action)
                    log_probs_batch.append(log_prob)
                    values_batch.append(value)

                # Convert continuous actions to discrete (argmax)
                discrete_actions = [np.argmax(a) for a in actions_batch]

                # Step environments
                next_states, rewards, dones, infos = vec_env.step(discrete_actions)

                # Store experiences
                for env_idx in range(n_envs):
                    buffer.add(
                        state=states[env_idx],
                        action=actions_batch[env_idx],
                        reward=rewards[env_idx],
                        value=values_batch[env_idx],
                        log_prob=log_probs_batch[env_idx],
                        done=dones[env_idx]
                    )

                    current_episode_reward[env_idx] += rewards[env_idx]
                    current_episode_length[env_idx] += 1

                    if dones[env_idx]:
                        episode_rewards.append(current_episode_reward[env_idx])
                        episode_lengths.append(current_episode_length[env_idx])
                        current_episode_reward[env_idx] = 0
                        current_episode_length[env_idx] = 0

                states = next_states
                total_steps += n_envs

            # Update agent
            update_info = agent.update(buffer)
            buffer.reset()
            n_updates += 1

            # Logging
            if total_steps % log_freq == 0:
                if len(episode_rewards) > 0:
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_length = np.mean(episode_lengths[-100:])
                else:
                    mean_reward = 0
                    mean_length = 0

                logger.info(
                    f"Steps: {total_steps}/{total_timesteps} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Mean Length: {mean_length:.0f} | "
                    f"Policy Loss: {update_info['policy_loss']:.4f} | "
                    f"Value Loss: {update_info['value_loss']:.4f}"
                )

            # Save checkpoint
            if total_steps % save_freq == 0:
                checkpoint_path = run_dir / f'ppo_checkpoint_{total_steps}.pt'
                agent.save(str(checkpoint_path))
                logger.info(f"Saved checkpoint: {checkpoint_path}")

                # Save training stats
                stats_df = pd.DataFrame({
                    'episode_reward': episode_rewards,
                    'episode_length': episode_lengths
                })
                stats_df.to_csv(run_dir / 'training_stats.csv', index=False)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_model_path = run_dir / 'ppo_final.pt'
    agent.save(str(final_model_path))
    logger.info(f"Saved final model: {final_model_path}")

    # Save final stats
    final_stats = pd.DataFrame({
        'episode_reward': episode_rewards,
        'episode_length': episode_lengths
    })
    final_stats.to_csv(run_dir / 'final_training_stats.csv', index=False)

    # Close environments
    vec_env.close()

    logger.info(f"Training complete! Total steps: {total_steps}")
    logger.info(f"Final mean reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

    return agent, episode_rewards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO agent with optimizations')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/dataset_with_regimes.csv',
                        help='Path to dataset')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--total-timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/ppo_optimized',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Train
    agent, rewards = train_ppo_optimized(
        data_path=args.data_path,
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )

    print(f"\nTraining summary:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
