"""
Proximal Policy Optimization (PPO) Agent for Continuous Portfolio Allocation

Implements PPO with:
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Vectorized environment support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'tanh'
    ):
        super(ActorCritic, self).__init__()

        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            self.activation,
            nn.LayerNorm(hidden_dims[0])
        )

        # Actor network (policy)
        actor_layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.actor_mean = nn.Sequential(
            *actor_layers,
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)  # Ensure valid portfolio weights
        )

        # Log std for continuous actions (learned parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network (value function)
        critic_layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.critic = nn.Sequential(
            *critic_layers,
            nn.Linear(prev_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns action distribution and value estimate."""
        shared_features = self.shared(state)

        # Actor: mean of action distribution
        action_mean = self.actor_mean(shared_features)

        # Critic: state value
        value = self.critic(shared_features)

        return action_mean, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, value = self.forward(state)

        if deterministic:
            return action_mean, torch.zeros_like(action_mean), value

        # Create diagonal Gaussian distribution
        std = self.log_std.exp()
        dist = Normal(action_mean, std)

        # Sample action
        action = dist.sample()

        # Normalize to ensure weights sum to 1
        action = torch.softmax(action, dim=-1)

        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_mean, values = self.forward(states)

        std = self.log_std.exp()
        dist = Normal(action_mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_probs, values, entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)

    def reset(self):
        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add experience to buffer."""
        self.states[self.ptr] = torch.FloatTensor(state)
        self.actions[self.ptr] = torch.FloatTensor(action)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def get(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Get all data with computed advantages using GAE."""
        buffer_size = self.buffer_size if self.full else self.ptr

        # Compute advantages using Generalized Advantage Estimation
        advantages = torch.zeros_like(self.rewards[:buffer_size])
        last_gae = 0

        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae

        # Returns are advantages + values
        returns = advantages + self.values[:buffer_size]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            self.states[:buffer_size].to(self.device),
            self.actions[:buffer_size].to(self.device),
            self.log_probs[:buffer_size].to(self.device),
            advantages.to(self.device),
            returns.to(self.device)
        )


class PPOAgent:
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        # Actor-Critic network
        self.ac_network = ActorCritic(
            state_dim, action_dim, hidden_dims
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.ac_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )

        # Training stats
        self.total_steps = 0
        self.update_count = 0

        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.ac_network.get_action(state_tensor, deterministic)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update policy using PPO."""
        # Get data from buffer
        states, actions, old_log_probs, advantages, returns = buffer.get(
            self.gamma, self.gae_lambda
        )

        buffer_size = states.shape[0]

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        n_updates = 0

        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Random permutation for mini-batches
            indices = torch.randperm(buffer_size)

            for start_idx in range(0, buffer_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions with current policy
                log_probs, values, entropy = self.ac_network.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss with clipped surrogate objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # KL divergence (for early stopping)
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - log_probs).mean().item()
                    total_kl_div += kl_div

                n_updates += 1

        self.update_count += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl_div / n_updates,
            'update_count': self.update_count
        }

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
        logger.info(f"Model loaded from {filepath}")
