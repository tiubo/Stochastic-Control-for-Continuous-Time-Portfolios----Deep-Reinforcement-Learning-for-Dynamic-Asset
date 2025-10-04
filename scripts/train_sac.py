"""
Training script for SAC agent on portfolio allocation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from agents.sac_agent import SACAgent
from environments.portfolio_env import PortfolioEnv


def train_sac(
    data_path: str = 'data/processed/dataset.csv',
    total_timesteps: int = 200000,
    eval_freq: int = 5000,
    save_freq: int = 10000,
    model_save_path: str = 'models/sac_portfolio.pth',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train SAC agent on portfolio allocation task.

    Args:
        data_path: Path to processed dataset
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
        model_save_path: Path to save model
        device: 'cpu' or 'cuda'
    """

    print("=" * 70)
    print("Training SAC Agent for Portfolio Allocation")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Data loaded: {len(data)} timesteps")

    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"Train: {len(train_data)} timesteps, Test: {len(test_data)} timesteps")

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

    print(f"\nEnvironment:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Assets: {env.n_assets}")

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
        buffer_capacity=1000000,
        batch_size=256,
        device=device
    )

    print(f"\nSAC Agent created")
    print(f"  Device: {device}")
    print(f"  Auto-tune temperature: True")

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num = 0

    rewards_history = []
    episode_rewards = []
    eval_returns = []
    losses_history = {'q1': [], 'q2': [], 'policy': [], 'alpha': []}

    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("-" * 70)

    pbar = tqdm(range(total_timesteps), desc="Training")

    for timestep in pbar:
        # Select action
        if timestep < 1000:
            # Random exploration for first 1000 steps
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward
        episode_length += 1
        rewards_history.append(reward)

        # Update agent
        if timestep >= 1000:
            losses = agent.update()

            if losses:
                losses_history['q1'].append(losses.get('q1_loss', 0))
                losses_history['q2'].append(losses.get('q2_loss', 0))
                losses_history['policy'].append(losses.get('policy_loss', 0))
                losses_history['alpha'].append(losses.get('alpha', 0.2))

        # Episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_num += 1

            pbar.set_postfix({
                'Episode': episode_num,
                'Reward': f'{episode_reward:.2f}',
                'Length': episode_length,
                'Alpha': f'{losses.get("alpha", 0.2):.3f}' if losses else 'N/A'
            })

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # Evaluation
        if (timestep + 1) % eval_freq == 0:
            eval_return = evaluate_agent(agent, test_data, n_episodes=3)
            eval_returns.append(eval_return)

            print(f"\nEvaluation at step {timestep + 1}:")
            print(f"  Average Return: {eval_return:.2f}")
            print(f"  Alpha: {losses.get('alpha', 0.2):.3f}" if losses else "  Alpha: N/A")
            print("-" * 70)

        # Save model
        if (timestep + 1) % save_freq == 0:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            checkpoint_path = model_save_path.replace('.pth', f'_step{timestep + 1}.pth')
            agent.save(checkpoint_path)

    # Final save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    # Plot training curves
    plot_training_curves(episode_rewards, eval_returns, losses_history, eval_freq)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    final_return = evaluate_agent(agent, test_data, n_episodes=10, verbose=True)
    print(f"\nFinal Average Return (10 episodes): {final_return:.2f}")

    return agent, episode_rewards, eval_returns


def evaluate_agent(agent, data, n_episodes=5, verbose=False):
    """
    Evaluate agent on test data.

    Args:
        agent: SAC agent
        data: Test data
        n_episodes: Number of evaluation episodes
        verbose: Print detailed info

    Returns:
        avg_return: Average return over episodes
    """
    env = PortfolioEnv(
        data=data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        action_type='continuous',
        reward_type='log_utility'
    )

    returns = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

        if verbose:
            print(f"  Episode {episode + 1}: Return = {episode_return:.2f}")

    avg_return = np.mean(returns)
    return avg_return


def plot_training_curves(episode_rewards, eval_returns, losses_history, eval_freq):
    """Plot training curves."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards) // 10)
        smoothed = pd.Series(episode_rewards).rolling(window).mean()
        axes[0, 0].plot(smoothed, linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)

    # Evaluation returns
    if eval_returns:
        eval_steps = np.arange(len(eval_returns)) * eval_freq
        axes[0, 1].plot(eval_steps, eval_returns, marker='o')
        axes[0, 1].set_title('Evaluation Returns')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Average Return')
        axes[0, 1].grid(True, alpha=0.3)

    # Q-losses
    if losses_history['q1']:
        axes[1, 0].plot(losses_history['q1'], alpha=0.5, label='Q1 Loss')
        axes[1, 0].plot(losses_history['q2'], alpha=0.5, label='Q2 Loss')

        if len(losses_history['q1']) > 100:
            window = len(losses_history['q1']) // 100
            q1_smooth = pd.Series(losses_history['q1']).rolling(window).mean()
            q2_smooth = pd.Series(losses_history['q2']).rolling(window).mean()
            axes[1, 0].plot(q1_smooth, linewidth=2, label='Q1 (smoothed)')
            axes[1, 0].plot(q2_smooth, linewidth=2, label='Q2 (smoothed)')

        axes[1, 0].set_title('Q-Network Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('MSE Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Alpha (temperature)
    if losses_history['alpha']:
        axes[1, 1].plot(losses_history['alpha'])
        axes[1, 1].set_title('Temperature Parameter (Alpha)')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Alpha')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sac_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to results/sac_training_curves.png")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SAC agent for portfolio allocation')
    parser.add_argument('--data-path', type=str, default='data/processed/dataset_with_regimes.csv',
                        help='Path to processed dataset')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                        help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=20000,
                        help='Model save frequency')
    parser.add_argument('--model-save-path', type=str, default='models/sac_trained.pth',
                        help='Path to save model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Train agent
    agent, episode_rewards, eval_returns = train_sac(
        data_path=args.data_path,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        model_save_path=args.model_save_path,
        device=args.device
    )

    print("\nTraining complete!")
