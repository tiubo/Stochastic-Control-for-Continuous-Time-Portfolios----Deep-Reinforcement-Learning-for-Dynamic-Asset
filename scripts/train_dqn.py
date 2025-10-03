"""
DQN Training Script
Train DQN agent on portfolio allocation task
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.environments.portfolio_env import PortfolioEnv
from src.agents.dqn_agent import DQNAgent
import argparse
from tqdm import tqdm


def train_dqn(
    data_path: str = "data/processed/complete_dataset.csv",
    n_episodes: int = 1000,
    save_path: str = "models/dqn_agent.pth",
    device: str = "cpu"
):
    """
    Train DQN agent on portfolio data.

    Args:
        data_path: Path to processed dataset
        n_episodes: Number of training episodes
        save_path: Path to save trained model
        device: 'cpu' or 'cuda'
    """
    print("Loading data...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Train/test split
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"Train: {len(train_data)} days, Test: {len(test_data)} days")

    # Create environment
    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        action_type='discrete',
        reward_type='log_utility'
    )

    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        device=device
    )

    print(f"\nTraining DQN Agent:")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Episodes: {n_episodes}")

    # Training loop
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()

            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # Log progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            metrics = env.get_portfolio_metrics()

            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward (50 eps): {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Final Portfolio Value: ${metrics['final_value']:,.2f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

        # Save checkpoint
        if (episode + 1) % 200 == 0:
            checkpoint_path = save_path.replace('.pth', f'_ep{episode+1}.pth')
            agent.save(checkpoint_path)

    # Save final model
    agent.save(save_path)
    print(f"\nTraining complete! Model saved to {save_path}")

    # Test on held-out data
    print("\nEvaluating on test data...")
    test_env = PortfolioEnv(
        data=test_data,
        initial_balance=100000.0,
        action_type='discrete',
        reward_type='log_utility'
    )

    state, _ = test_env.reset()
    done = False
    test_reward = 0

    while not done:
        action = agent.select_action(state, epsilon=0.0)  # Greedy
        state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        test_reward += reward

    test_metrics = test_env.get_portfolio_metrics()

    print("\nTest Results:")
    print(f"Total Reward: {test_reward:.3f}")
    print(f"Total Return: {test_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown']:.2%}")
    print(f"Final Value: ${test_metrics['final_value']:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--data", type=str, default="data/processed/complete_dataset.csv",
                        help="Path to processed dataset")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--save", type=str, default="models/dqn_agent.pth",
                        help="Path to save model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")

    args = parser.parse_args()

    train_dqn(
        data_path=args.data,
        n_episodes=args.episodes,
        save_path=args.save,
        device=args.device
    )
