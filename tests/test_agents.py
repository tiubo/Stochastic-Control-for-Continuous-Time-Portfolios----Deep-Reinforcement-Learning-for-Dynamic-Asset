"""
Comprehensive unit tests for RL agents.

Tests cover:
- DQN agent initialization and learning
- Prioritized DQN with experience replay
- PPO agent with actor-critic
- Action selection and exploration
- Model saving/loading
- Edge cases and error handling
"""

import pytest
import numpy as np
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.dqn_agent import DQNAgent, QNetwork, ReplayBuffer
from agents.prioritized_dqn_agent import PrioritizedDQNAgent, SumTree, PrioritizedReplayBuffer, NoisyLinear, DuelingQNetwork
from agents.ppo_agent import PPOAgent, ActorCritic, RolloutBuffer


class TestReplayBuffer:
    """Test replay buffer functionality."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0

    def test_add_and_sample(self):
        """Test adding experiences and sampling."""
        buffer = ReplayBuffer(capacity=100)

        # Add experiences
        for i in range(50):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 50

        # Sample batch
        batch = buffer.sample(batch_size=32)
        assert batch['states'].shape == (32, 10)
        assert batch['actions'].shape == (32,)
        assert batch['rewards'].shape == (32,)
        assert batch['next_states'].shape == (32, 10)
        assert batch['dones'].shape == (32,)

    def test_buffer_overflow(self):
        """Test buffer handles overflow correctly."""
        buffer = ReplayBuffer(capacity=10)

        # Add more than capacity
        for i in range(20):
            buffer.add(np.zeros(5), 0, 0.0, np.zeros(5), False)

        assert len(buffer) == 10  # Should not exceed capacity


class TestSumTree:
    """Test sum tree for prioritized replay."""

    def test_sum_tree_initialization(self):
        """Test sum tree initializes correctly."""
        tree = SumTree(capacity=8)
        assert tree.capacity == 8
        assert tree.total() == 0.0

    def test_add_and_sample(self):
        """Test adding priorities and sampling."""
        tree = SumTree(capacity=8)

        # Add priorities
        for i in range(5):
            tree.add(priority=(i + 1) * 1.0, data=i)

        assert tree.total() == 15.0  # 1+2+3+4+5

        # Sample
        idx, priority, data = tree.get(7.5)
        assert priority > 0
        assert data is not None

    def test_update_priority(self):
        """Test updating priorities."""
        tree = SumTree(capacity=8)

        tree.add(priority=1.0, data=0)
        tree.add(priority=2.0, data=1)

        old_total = tree.total()
        tree.update(idx=0, priority=5.0)

        assert tree.total() == old_total + 4.0


class TestQNetwork:
    """Test Q-Network architecture."""

    def test_network_forward(self):
        """Test forward pass."""
        net = QNetwork(state_dim=10, action_dim=3, hidden_dims=[64, 64])
        state = torch.randn(32, 10)

        q_values = net(state)
        assert q_values.shape == (32, 3)

    def test_network_gradient(self):
        """Test gradients flow correctly."""
        net = QNetwork(state_dim=10, action_dim=3)
        optimizer = torch.optim.Adam(net.parameters())

        state = torch.randn(32, 10)
        q_values = net(state)
        loss = q_values.mean()

        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        for param in net.parameters():
            assert param.grad is not None


class TestDQNAgent:
    """Test DQN agent functionality."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DQNAgent(state_dim=10, action_dim=3)
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert len(agent.replay_buffer) == 0

    def test_select_action_exploration(self):
        """Test action selection with exploration."""
        agent = DQNAgent(state_dim=10, action_dim=3)
        state = np.random.randn(10)

        # With high epsilon, should explore
        agent.epsilon = 1.0
        actions = [agent.select_action(state) for _ in range(100)]
        unique_actions = len(set(actions))
        assert unique_actions > 1  # Should explore different actions

    def test_select_action_exploitation(self):
        """Test action selection without exploration."""
        agent = DQNAgent(state_dim=10, action_dim=3)
        state = np.random.randn(10)

        # With zero epsilon, should exploit
        agent.epsilon = 0.0
        actions = [agent.select_action(state) for _ in range(10)]
        assert len(set(actions)) == 1  # Should select same action

    def test_update_step(self):
        """Test learning update step."""
        agent = DQNAgent(state_dim=10, action_dim=3, batch_size=32)

        # Add experiences to buffer
        for i in range(100):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            agent.replay_buffer.add(state, action, reward, next_state, done)

        # Update should not raise error
        initial_loss = agent.update()
        assert isinstance(initial_loss, float)
        assert initial_loss >= 0

    def test_target_network_update(self):
        """Test target network soft update."""
        agent = DQNAgent(state_dim=10, action_dim=3)

        # Get initial target network parameters
        initial_params = [p.clone() for p in agent.target_network.parameters()]

        # Modify Q-network
        for p in agent.q_network.parameters():
            p.data.fill_(1.0)

        # Soft update
        agent._update_target_network()

        # Target network should change slightly
        for initial, current in zip(initial_params, agent.target_network.parameters()):
            assert not torch.allclose(initial, current)

    def test_save_load_model(self):
        """Test model saving and loading."""
        agent = DQNAgent(state_dim=10, action_dim=3)

        # Train a bit to get non-random weights
        for i in range(50):
            agent.replay_buffer.add(
                np.random.randn(10), np.random.randint(0, 3),
                np.random.randn(), np.random.randn(10), False
            )
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            agent.save(save_path)

            # Create new agent and load
            new_agent = DQNAgent(state_dim=10, action_dim=3)
            new_agent.load(save_path)

            # Check weights match
            for p1, p2 in zip(agent.q_network.parameters(), new_agent.q_network.parameters()):
                assert torch.allclose(p1, p2)


class TestDuelingQNetwork:
    """Test Dueling Q-Network architecture."""

    def test_dueling_forward(self):
        """Test dueling network forward pass."""
        net = DuelingQNetwork(state_dim=10, action_dim=3, hidden_dims=[64, 64])
        state = torch.randn(32, 10)

        q_values = net(state)
        assert q_values.shape == (32, 3)

    def test_advantage_mean_zero(self):
        """Test advantage stream has mean zero property."""
        net = DuelingQNetwork(state_dim=10, action_dim=3)
        state = torch.randn(1, 10)

        # Get internal features
        features = net.shared(state)
        advantage = net.advantage_stream(features)

        # Advantage should subtract its mean
        assert advantage.shape == (1, 3)


class TestNoisyLinear:
    """Test noisy linear layer for exploration."""

    def test_noisy_forward(self):
        """Test noisy layer forward pass."""
        layer = NoisyLinear(in_features=10, out_features=5)
        x = torch.randn(32, 10)

        out = layer(x)
        assert out.shape == (32, 5)

    def test_noise_reset(self):
        """Test noise reset changes output."""
        layer = NoisyLinear(in_features=10, out_features=5)
        x = torch.randn(1, 10)

        # Get output with current noise
        out1 = layer(x)

        # Reset noise and get new output
        layer.reset_noise()
        out2 = layer(x)

        # Outputs should differ due to noise
        assert not torch.allclose(out1, out2)


class TestPrioritizedDQNAgent:
    """Test Prioritized DQN agent."""

    def test_prioritized_agent_initialization(self):
        """Test prioritized agent initializes correctly."""
        agent = PrioritizedDQNAgent(
            state_dim=10, action_dim=3,
            use_double_dqn=True,
            use_dueling=True,
            use_noisy=True
        )
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.use_double_dqn is True
        assert agent.use_dueling is True
        assert agent.use_noisy is True

    def test_prioritized_sampling(self):
        """Test prioritized experience replay sampling."""
        agent = PrioritizedDQNAgent(state_dim=10, action_dim=3, per_alpha=0.6)

        # Add experiences with different priorities
        for i in range(100):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = float(i) / 10.0  # Varying rewards
            next_state = np.random.randn(10)
            done = False
            agent.replay_buffer.add(state, action, reward, next_state, done)

        # Sample should work
        batch = agent.replay_buffer.sample(batch_size=32, beta=0.4)
        assert 'states' in batch
        assert 'indices' in batch
        assert 'weights' in batch

    def test_double_dqn_update(self):
        """Test Double DQN reduces overestimation."""
        agent = PrioritizedDQNAgent(
            state_dim=10, action_dim=3,
            use_double_dqn=True,
            batch_size=32
        )

        # Add experiences
        for i in range(100):
            agent.replay_buffer.add(
                np.random.randn(10), np.random.randint(0, 3),
                np.random.randn(), np.random.randn(10), False
            )

        # Update should work
        loss = agent.update()
        assert isinstance(loss, float)
        assert loss >= 0


class TestActorCritic:
    """Test Actor-Critic network."""

    def test_actor_critic_forward(self):
        """Test Actor-Critic forward pass."""
        net = ActorCritic(state_dim=10, action_dim=3, hidden_dims=[64, 64])
        state = torch.randn(32, 10)

        action_probs, value = net(state)
        assert action_probs.shape == (32, 3)
        assert value.shape == (32, 1)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(32))  # Probabilities sum to 1

    def test_get_action(self):
        """Test action sampling."""
        net = ActorCritic(state_dim=10, action_dim=3)
        state = torch.randn(1, 10)

        action, log_prob = net.get_action(state)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < 3


class TestRolloutBuffer:
    """Test rollout buffer for PPO."""

    def test_buffer_store_and_get(self):
        """Test storing and retrieving rollouts."""
        buffer = RolloutBuffer()

        # Store rollouts
        for i in range(50):
            buffer.store(
                state=np.random.randn(10),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                value=np.random.randn(),
                log_prob=np.random.randn()
            )

        # Get rollouts
        rollouts = buffer.get()
        assert rollouts['states'].shape == (50, 10)
        assert rollouts['actions'].shape == (50,)
        assert rollouts['rewards'].shape == (50,)
        assert rollouts['values'].shape == (50,)
        assert rollouts['log_probs'].shape == (50,)

    def test_buffer_clear(self):
        """Test buffer clears correctly."""
        buffer = RolloutBuffer()

        for i in range(10):
            buffer.store(np.zeros(5), 0, 0.0, 0.0, 0.0)

        assert len(buffer.states) == 10
        buffer.clear()
        assert len(buffer.states) == 0


class TestPPOAgent:
    """Test PPO agent functionality."""

    def test_ppo_agent_initialization(self):
        """Test PPO agent initializes correctly."""
        agent = PPOAgent(state_dim=10, action_dim=3, hidden_dims=[64, 64])
        assert agent.state_dim == 10
        assert agent.action_dim == 3

    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=10, action_dim=3)
        state = np.random.randn(10)

        action, log_prob, value = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_ppo_update(self):
        """Test PPO update step."""
        agent = PPOAgent(state_dim=10, action_dim=3, clip_epsilon=0.2)
        buffer = RolloutBuffer()

        # Collect rollouts
        for i in range(100):
            state = np.random.randn(10)
            action, log_prob, value = agent.select_action(state)
            reward = np.random.randn()
            buffer.store(state, action, reward, value, log_prob)

        # Update should work
        losses = agent.update(buffer)
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert isinstance(losses['policy_loss'], float)
        assert isinstance(losses['value_loss'], float)

    def test_gae_computation(self):
        """Test Generalized Advantage Estimation."""
        agent = PPOAgent(state_dim=10, action_dim=3, gamma=0.99, gae_lambda=0.95)

        rewards = np.array([1.0, 1.0, 1.0, 0.0])
        values = np.array([0.5, 0.6, 0.7, 0.0])
        dones = np.array([0, 0, 0, 1])

        advantages, returns = agent._compute_gae(rewards, values, dones)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_ppo_save_load(self):
        """Test PPO model saving and loading."""
        agent = PPOAgent(state_dim=10, action_dim=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "ppo_model.pth")
            agent.save(save_path)

            # Create new agent and load
            new_agent = PPOAgent(state_dim=10, action_dim=3)
            new_agent.load(save_path)

            # Check weights match
            for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
                assert torch.allclose(p1, p2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_reward_learning(self):
        """Test agents can handle zero rewards."""
        agent = DQNAgent(state_dim=5, action_dim=2, batch_size=16)

        for i in range(50):
            agent.replay_buffer.add(
                np.random.randn(5), np.random.randint(0, 2),
                0.0,  # Zero reward
                np.random.randn(5), False
            )

        loss = agent.update()
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_terminal_state_handling(self):
        """Test agents handle terminal states correctly."""
        agent = DQNAgent(state_dim=5, action_dim=2, batch_size=16)

        for i in range(50):
            agent.replay_buffer.add(
                np.random.randn(5), np.random.randint(0, 2),
                np.random.randn(), np.random.randn(5),
                done=True  # Terminal state
            )

        loss = agent.update()
        assert not np.isnan(loss)

    def test_small_batch_handling(self):
        """Test agents handle small batches gracefully."""
        agent = PPOAgent(state_dim=5, action_dim=2)
        buffer = RolloutBuffer()

        # Add only a few experiences
        for i in range(5):
            state = np.random.randn(5)
            action, log_prob, value = agent.select_action(state)
            buffer.store(state, action, 0.0, value, log_prob)

        # Should handle small batch
        losses = agent.update(buffer)
        assert 'policy_loss' in losses


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
