"""
RL Strategy Adapters for Backtesting Engine

Connects trained RL agents (DQN, PPO, SAC) to the BacktestEngine
using the Strategy interface.
"""

import numpy as np
import torch
from typing import Dict, Any
from pathlib import Path

from .backtest_engine import Strategy
from ..agents.dqn_agent import DQNAgent
from ..agents.ppo_agent import PPOAgent
from ..agents.sac_agent import SACAgent


class DQNStrategyAdapter(Strategy):
    """
    Adapter for trained DQN agent.

    DQN outputs discrete actions (0, 1, 2) which map to:
    - 0: Conservative (low stocks, high bonds)
    - 1: Balanced
    - 2: Aggressive (high stocks, low bonds)
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained DQN model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path

        # Action mappings (will be customized based on asset count)
        # For 4 assets: SPY, TLT, GLD, BTC
        self.action_mappings = {
            0: np.array([0.20, 0.50, 0.20, 0.10]),  # Conservative: bonds-heavy
            1: np.array([0.40, 0.30, 0.20, 0.10]),  # Balanced
            2: np.array([0.60, 0.10, 0.20, 0.10]),  # Aggressive: stocks-heavy
        }

        # Load model
        self.agent = None
        self._load_model()

    def _load_model(self):
        """Load trained DQN agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"DQN model not found: {self.model_path}")

        # Initialize agent (state_dim and action_dim will be set from checkpoint)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)  # Default from our env
        action_dim = checkpoint.get('action_dim', 3)

        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.policy_net.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained DQN agent.

        Args:
            data: Market data DataFrame
            current_idx: Current time index
            current_weights: Current portfolio weights (not used by DQN)
            **kwargs: Additional arguments

        Returns:
            weights: Portfolio weights (sums to 1)
        """
        # Extract state from current market data
        state = self._extract_state(data, current_idx)

        # Get action from agent
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.agent.select_action(state_tensor, epsilon=0.0)  # Greedy

        # Map action to weights
        weights = self.action_mappings.get(action, self.action_mappings[1])

        return weights

    def _extract_state(self, data, idx) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PortfolioEnv._get_state().

        State components:
        - Portfolio weights (n_assets)
        - Recent returns (5 days × n_assets)
        - Current volatility (n_assets)
        - Market regime one-hot (3)
        - VIX (1)
        - Treasury rate (1)
        - Portfolio value normalized (1)
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        n_assets = len(return_cols)

        state_components = []

        # 1. Current portfolio weights (zeros for initial allocation)
        weights = np.zeros(n_assets)  # DQN doesn't use current weights
        state_components.append(weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized to 1.0 for backtest)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)


class PPOStrategyAdapter(Strategy):
    """
    Adapter for trained PPO agent.
    PPO outputs continuous weights directly.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained PPO model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.agent = None
        self._load_model()

    def _load_model(self):
        """Load trained PPO agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"PPO model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)
        action_dim = checkpoint.get('action_dim', 4)  # Continuous weights

        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.actor.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained PPO agent.

        Args:
            data: Market data DataFrame
            current_idx: Current time index
            current_weights: Current portfolio weights
            **kwargs: Additional arguments

        Returns:
            weights: Portfolio weights (sums to 1, each in [0, 1])
        """
        # Extract state
        state = self._extract_state(data, current_idx, current_weights)

        # Get action from agent
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _, _ = self.agent.select_action(state_tensor, training=False)
            action = action.cpu().numpy().flatten()

        # Convert action to weights (apply softmax for normalization)
        weights = np.exp(action) / np.sum(np.exp(action))
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()  # Ensure sum to 1

        return weights

    def _extract_state(self, data, idx, current_weights) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PortfolioEnv._get_state().
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        n_assets = len(return_cols)

        state_components = []

        # 1. Current portfolio weights (use actual weights)
        state_components.append(current_weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized to 1.0 for backtest)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)


class SACStrategyAdapter(Strategy):
    """
    Adapter for trained SAC agent.
    SAC outputs continuous weights directly.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained SAC model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.agent = None
        self._load_model()

    def _load_model(self):
        """Load trained SAC agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SAC model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)
        action_dim = checkpoint.get('action_dim', 4)

        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.actor.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained SAC agent.

        Args:
            data: Market data DataFrame
            current_idx: Current time index
            current_weights: Current portfolio weights
            **kwargs: Additional arguments

        Returns:
            weights: Portfolio weights (sums to 1, each in [0, 1])
        """
        # Extract state
        state = self._extract_state(data, current_idx, current_weights)

        # Get action from agent (deterministic for evaluation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.agent.select_action(state_tensor, evaluate=True)
            action = action.cpu().numpy().flatten()

        # Convert action to weights (apply softmax)
        weights = np.exp(action) / np.sum(np.exp(action))
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()

        return weights

    def _extract_state(self, data, idx, current_weights) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PortfolioEnv._get_state().
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        n_assets = len(return_cols)

        state_components = []

        # 1. Current portfolio weights (use actual weights)
        state_components.append(current_weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized to 1.0 for backtest)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)


def create_rl_strategy_adapter(agent_type: str, model_path: str, device: str = 'cpu') -> Strategy:
    """
    Factory function to create RL strategy adapters.

    Args:
        agent_type: 'dqn', 'ppo', or 'sac'
        model_path: Path to trained model
        device: 'cpu' or 'cuda'

    Returns:
        Strategy adapter instance

    Example:
        >>> strategy = create_rl_strategy_adapter('dqn', 'models/dqn_trained.pth')
        >>> engine = BacktestEngine(config)
        >>> results = engine.run(strategy, data, returns)
    """
    adapters = {
        'dqn': DQNStrategyAdapter,
        'ppo': PPOStrategyAdapter,
        'sac': SACStrategyAdapter,
    }

    agent_type = agent_type.lower()
    if agent_type not in adapters:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(adapters.keys())}")

    return adapters[agent_type](model_path=model_path, device=device)
