"""
Prioritized Experience Replay DQN Agent for Portfolio Allocation

Implements:
- Prioritized Experience Replay (PER)
- Double DQN (DDQN)
- Dueling DQN architecture
- Noisy Networks for exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Stores priorities in a binary tree for O(log n) updates and sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience with priority."""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority of experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """Get experience based on priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling exponent
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def _get_priority(self, td_error: float) -> float:
        """Calculate priority from TD error."""
        return (abs(td_error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done, td_error: float = 1.0):
        """Add experience with priority."""
        priority = self._get_priority(td_error)
        experience = (state, action, reward, next_state, done)
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritized sampling."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Annealing beta for importance sampling
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            if data is None:
                continue

            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        # Calculate importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -beta)
        is_weights /= is_weights.max()  # Normalize

        # Unpack batch
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones, indices, is_weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        use_noisy: bool = True
    ):
        super(DuelingQNetwork, self).__init__()
        self.action_dim = action_dim
        self.use_noisy = use_noisy

        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0])
        )

        # Value stream
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                NoisyLinear(hidden_dims[1], 1)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1)
            )

        # Advantage stream
        if use_noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                NoisyLinear(hidden_dims[1], action_dim)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], action_dim)
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Reset noise for noisy layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedDQNAgent:
    """Prioritized DQN Agent with Double DQN and Dueling architecture."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        use_noisy: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_noisy = use_noisy
        self.device = device

        # Q-Networks
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, hidden_dims, use_noisy
        ).to(device)

        self.target_network = DuelingQNetwork(
            state_dim, action_dim, hidden_dims, use_noisy
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=100000
        )

        # Training stats
        self.total_steps = 0
        self.update_count = 0

        logger.info(f"Prioritized DQN initialized: Double={use_double_dqn}, Dueling={use_dueling}, Noisy={use_noisy}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy (or noisy networks)."""
        # Noisy networks provide exploration, so epsilon can be 0
        if self.use_noisy:
            epsilon = 0.0

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update(self) -> Dict[str, float]:
        """Update Q-network using prioritized experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD errors for priority update
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # Loss with importance sampling weights
        loss = (is_weights * (current_q - target_q).pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Reset noise
        if self.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.total_steps += 1
        self.update_count += 1

        return {
            'loss': loss.item(),
            'mean_q': current_q.mean().item(),
            'mean_td_error': abs(td_errors).mean(),
            'update_count': self.update_count
        }

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
        logger.info(f"Model loaded from {filepath}")
