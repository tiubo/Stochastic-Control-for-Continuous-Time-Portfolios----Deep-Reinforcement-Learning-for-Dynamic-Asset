"""
Master Training Script for All RL Agents

Trains DQN, PPO, and SAC agents sequentially with:
- Proper logging and checkpointing
- Training progress visualization
- Performance evaluation
- Model saving
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import json

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.environments.portfolio_env import PortfolioEnv


class TrainingLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str, agent_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name
        self.metrics = {
            'episode': [],
            'episode_reward': [],
            'episode_length': [],
            'avg_reward_100': [],
            'portfolio_value': [],
            'sharpe_ratio': []
        }

    def log(self, episode: int, reward: float, length: int,
            portfolio_value: float, sharpe: float = 0.0):
        """Log training metrics."""
        self.metrics['episode'].append(episode)
        self.metrics['episode_reward'].append(reward)
        self.metrics['episode_length'].append(length)
        self.metrics['portfolio_value'].append(portfolio_value)
        self.metrics['sharpe_ratio'].append(sharpe)

        # Calculate rolling average
        recent_rewards = self.metrics['episode_reward'][-100:]
        avg_reward = np.mean(recent_rewards)
        self.metrics['avg_reward_100'].append(avg_reward)

    def save(self):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_dir / f'{self.agent_name}_training_log.csv', index=False)

    def plot(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.metrics['episode'], self.metrics['episode_reward'],
                alpha=0.3, label='Episode Reward')
        ax.plot(self.metrics['episode'], self.metrics['avg_reward_100'],
                linewidth=2, label='MA-100')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{self.agent_name} - Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Portfolio value
        ax = axes[0, 1]
        ax.plot(self.metrics['episode'], self.metrics['portfolio_value'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title(f'{self.agent_name} - Portfolio Value')
        ax.grid(True, alpha=0.3)

        # Sharpe ratio
        ax = axes[1, 0]
        ax.plot(self.metrics['episode'], self.metrics['sharpe_ratio'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'{self.agent_name} - Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[1, 1]
        ax.plot(self.metrics['episode'], self.metrics['episode_length'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title(f'{self.agent_name} - Episode Length')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe


def train_dqn(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    total_episodes: int = 1000,
    output_dir: str = 'models/dqn'
):
    """Train DQN agent."""

    print("\n" + "="*80)
    print("TRAINING DQN AGENT")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        action_type='discrete',
        reward_type='log_utility'
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64
    )

    # Training loop
    logger = TrainingLogger(output_dir, 'DQN')

    for episode in tqdm(range(total_episodes), desc="DQN Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_returns = []

        done = False
        while not done:
            # Select action
            action = agent.select_action(state)

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            agent.train()

            # Update
            state = next_state
            episode_reward += reward
            episode_length += 1

            if 'return' in info:
                episode_returns.append(info['return'])

        # Calculate metrics
        portfolio_value = env.portfolio_value
        sharpe = calculate_sharpe_ratio(np.array(episode_returns)) if episode_returns else 0.0

        # Log
        logger.log(episode, episode_reward, episode_length, portfolio_value, sharpe)

        # Save checkpoints
        if (episode + 1) % 100 == 0:
            agent.save(output_dir / f'dqn_episode_{episode+1}.pth')
            logger.save()

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = logger.metrics['avg_reward_100'][-1]
            print(f"\nEpisode {episode+1}/{total_episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Portfolio Value: ${portfolio_value:,.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")

    # Save final model
    agent.save(output_dir / 'dqn_final.pth')
    logger.save()
    logger.plot(output_dir / 'dqn_training_curves.png')

    print(f"\nDQN training complete! Model saved to {output_dir}")

    return agent, logger


def train_ppo(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    total_timesteps: int = 500000,
    output_dir: str = 'models/ppo'
):
    """Train PPO agent."""

    print("\n" + "="*80)
    print("TRAINING PPO AGENT")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        action_type='continuous',
        reward_type='log_utility'
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64
    )

    # Training loop
    logger = TrainingLogger(output_dir, 'PPO')

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_returns = []
    episode = 0

    for timestep in tqdm(range(total_timesteps), desc="PPO Training"):
        # Select action
        action, log_prob = agent.select_action(state)

        # Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(state, action, reward, log_prob, done)

        # Update
        state = next_state
        episode_reward += reward
        episode_length += 1

        if 'return' in info:
            episode_returns.append(info['return'])

        # Episode end
        if done:
            # Calculate metrics
            portfolio_value = env.portfolio_value
            sharpe = calculate_sharpe_ratio(np.array(episode_returns)) if episode_returns else 0.0

            # Log
            logger.log(episode, episode_reward, episode_length, portfolio_value, sharpe)

            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            episode += 1

            # Print progress
            if episode % 50 == 0:
                avg_reward = logger.metrics['avg_reward_100'][-1] if len(logger.metrics['avg_reward_100']) > 0 else 0
                print(f"\nEpisode {episode}, Timestep {timestep}/{total_timesteps}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Portfolio Value: ${portfolio_value:,.2f}")

        # Update policy
        if (timestep + 1) % 2048 == 0:
            agent.update()

        # Save checkpoints
        if (timestep + 1) % 50000 == 0:
            agent.save(output_dir / f'ppo_timestep_{timestep+1}.pth')
            logger.save()

    # Save final model
    agent.save(output_dir / 'ppo_final.pth')
    logger.save()
    logger.plot(output_dir / 'ppo_training_curves.png')

    print(f"\nPPO training complete! Model saved to {output_dir}")

    return agent, logger


def train_sac(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    total_timesteps: int = 500000,
    output_dir: str = 'models/sac'
):
    """Train SAC agent."""

    print("\n" + "="*80)
    print("TRAINING SAC AGENT")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        action_type='continuous',
        reward_type='log_utility'
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True,
        buffer_capacity=100000,
        batch_size=256
    )

    # Training loop
    logger = TrainingLogger(output_dir, 'SAC')

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_returns = []
    episode = 0

    for timestep in tqdm(range(total_timesteps), desc="SAC Training"):
        # Select action
        action = agent.select_action(state, evaluate=False)

        # Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(state, action, reward, next_state, done)

        # Train (after initial exploration)
        if timestep > 1000:
            agent.update()

        # Update
        state = next_state
        episode_reward += reward
        episode_length += 1

        if 'return' in info:
            episode_returns.append(info['return'])

        # Episode end
        if done:
            # Calculate metrics
            portfolio_value = env.portfolio_value
            sharpe = calculate_sharpe_ratio(np.array(episode_returns)) if episode_returns else 0.0

            # Log
            logger.log(episode, episode_reward, episode_length, portfolio_value, sharpe)

            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            episode += 1

            # Print progress
            if episode % 50 == 0:
                avg_reward = logger.metrics['avg_reward_100'][-1] if len(logger.metrics['avg_reward_100']) > 0 else 0
                print(f"\nEpisode {episode}, Timestep {timestep}/{total_timesteps}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Portfolio Value: ${portfolio_value:,.2f}")
                print(f"  Alpha: {agent.alpha:.4f}")

        # Save checkpoints
        if (timestep + 1) % 50000 == 0:
            agent.save(output_dir / f'sac_timestep_{timestep+1}.pth')
            logger.save()

    # Save final model
    agent.save(output_dir / 'sac_final.pth')
    logger.save()
    logger.plot(output_dir / 'sac_training_curves.png')

    print(f"\nSAC training complete! Model saved to {output_dir}")

    return agent, logger


def main():
    """Main training function."""

    print("="*80)
    print("MASTER TRAINING SCRIPT - ALL RL AGENTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    data_path = 'data/processed/complete_dataset.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"Data loaded: {len(data)} timesteps")
    print(f"Train: {len(train_data)} timesteps")
    print(f"Test: {len(test_data)} timesteps")

    # Training configuration
    config = {
        'dqn_episodes': 1000,
        'ppo_timesteps': 500000,
        'sac_timesteps': 500000
    }

    print("\nTraining configuration:")
    print(json.dumps(config, indent=2))

    # Train all agents
    results = {}

    # 1. DQN
    dqn_agent, dqn_logger = train_dqn(train_data, test_data,
                                       total_episodes=config['dqn_episodes'])
    results['DQN'] = dqn_logger.metrics

    # 2. PPO
    ppo_agent, ppo_logger = train_ppo(train_data, test_data,
                                       total_timesteps=config['ppo_timesteps'])
    results['PPO'] = ppo_logger.metrics

    # 3. SAC
    sac_agent, sac_logger = train_sac(train_data, test_data,
                                       total_timesteps=config['sac_timesteps'])
    results['SAC'] = sac_logger.metrics

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)

    for agent_name, metrics in results.items():
        final_reward = metrics['avg_reward_100'][-1] if metrics['avg_reward_100'] else 0
        final_value = metrics['portfolio_value'][-1] if metrics['portfolio_value'] else 0
        final_sharpe = metrics['sharpe_ratio'][-1] if metrics['sharpe_ratio'] else 0

        print(f"\n{agent_name}:")
        print(f"  Final Avg Reward (MA-100): {final_reward:.2f}")
        print(f"  Final Portfolio Value: ${final_value:,.2f}")
        print(f"  Final Sharpe Ratio: {final_sharpe:.3f}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll models saved to models/")
    print("Training logs saved to respective directories")


if __name__ == "__main__":
    main()
