"""
Optimized Prioritized DQN Training Script

Features:
- Prioritized Experience Replay
- Double DQN
- Dueling architecture
- Noisy Networks
- Automatic checkpointing
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
from collections import deque

from src.agents.prioritized_dqn_agent import PrioritizedDQNAgent
from src.environments.portfolio_env import PortfolioEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_prioritized_dqn(
    data_path: str = 'data/processed/dataset_with_regimes.csv',
    total_timesteps: int = 500000,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    buffer_capacity: int = 100000,
    batch_size: int = 64,
    target_update_freq: int = 1000,
    use_double_dqn: bool = True,
    use_noisy: bool = True,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    save_freq: int = 50000,
    log_freq: int = 1000,
    eval_freq: int = 10000,
    output_dir: str = 'models/prioritized_dqn',
    device: str = 'cpu'
):
    """Train Prioritized DQN agent."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Prioritized DQN training: {run_dir}")
    logger.info(f"Using device: {device}")

    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded dataset: {data.shape}")

    # Create environment
    env = PortfolioEnv(
        data=data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        action_type='discrete',
        reward_type='log_utility'
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PrioritizedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        use_double_dqn=use_double_dqn,
        use_noisy=use_noisy,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        device=device
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    losses = []
    q_values = []

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    epsilon = epsilon_start if not use_noisy else 0.0

    recent_rewards = deque(maxlen=100)

    logger.info("Starting training loop...")

    try:
        for step in range(total_timesteps):
            # Select action
            action = agent.select_action(state, epsilon)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done, td_error=abs(reward))

            episode_reward += reward
            episode_length += 1

            # Update agent
            if len(agent.replay_buffer) >= batch_size:
                update_info = agent.update()

                if update_info:
                    losses.append(update_info['loss'])
                    q_values.append(update_info['mean_q'])

            # Episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                recent_rewards.append(episode_reward)

                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                # Decay epsilon
                if not use_noisy:
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)
            else:
                state = next_state

            # Logging
            if (step + 1) % log_freq == 0:
                mean_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
                mean_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
                mean_q = np.mean(q_values[-100:]) if len(q_values) > 0 else 0

                logger.info(
                    f"Steps: {step+1}/{total_timesteps} | "
                    f"Episodes: {len(episode_rewards)} | "
                    f"Mean Reward (100): {mean_reward:.2f} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Mean Q: {mean_q:.2f} | "
                    f"Buffer: {len(agent.replay_buffer)}"
                )

            # Evaluation
            if (step + 1) % eval_freq == 0:
                eval_reward = evaluate_agent(agent, env, n_episodes=10)
                eval_rewards.append(eval_reward)
                logger.info(f"Evaluation reward (10 episodes): {eval_reward:.2f}")

            # Save checkpoint
            if (step + 1) % save_freq == 0:
                checkpoint_path = run_dir / f'dqn_checkpoint_{step+1}.pt'
                agent.save(str(checkpoint_path))
                logger.info(f"Saved checkpoint: {checkpoint_path}")

                # Save training stats
                stats_df = pd.DataFrame({
                    'episode_reward': episode_rewards,
                    'episode_length': episode_lengths
                })
                stats_df.to_csv(run_dir / 'training_stats.csv', index=False)

                if len(eval_rewards) > 0:
                    eval_df = pd.DataFrame({'eval_reward': eval_rewards})
                    eval_df.to_csv(run_dir / 'eval_stats.csv', index=False)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_model_path = run_dir / 'dqn_final.pt'
    agent.save(str(final_model_path))
    logger.info(f"Saved final model: {final_model_path}")

    # Save final stats
    final_stats = pd.DataFrame({
        'episode_reward': episode_rewards,
        'episode_length': episode_lengths
    })
    final_stats.to_csv(run_dir / 'final_training_stats.csv', index=False)

    # Close environment
    env.close()

    logger.info(f"Training complete! Total steps: {total_timesteps}")
    logger.info(f"Total episodes: {len(episode_rewards)}")
    logger.info(f"Final mean reward (last 100 episodes): {np.mean(recent_rewards):.2f}")

    return agent, episode_rewards


def evaluate_agent(agent, env, n_episodes: int = 10) -> float:
    """Evaluate agent performance."""
    eval_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Prioritized DQN agent')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/dataset_with_regimes.csv',
                        help='Path to dataset')
    parser.add_argument('--total-timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--buffer-capacity', type=int, default=100000,
                        help='Replay buffer capacity')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--no-double-dqn', action='store_true',
                        help='Disable Double DQN')
    parser.add_argument('--no-noisy', action='store_true',
                        help='Disable Noisy Networks')
    parser.add_argument('--output-dir', type=str, default='models/prioritized_dqn',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Train
    agent, rewards = train_prioritized_dqn(
        data_path=args.data_path,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        use_double_dqn=not args.no_double_dqn,
        use_noisy=not args.no_noisy,
        output_dir=args.output_dir,
        device=args.device
    )

    print(f"\nTraining summary:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
