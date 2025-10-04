"""
Main configuration module for Deep RL Portfolio Allocation
Pure Python configuration to replace YAML files
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

@dataclass
class ProjectConfig:
    """Project metadata."""
    name: str = "Deep RL Portfolio Allocation"
    version: str = "1.0.0"
    description: str = "Multi-agent RL system for dynamic asset allocation"


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_path: str = "data/processed/dataset.csv"

    # Asset tickers
    assets: List[str] = field(default_factory=lambda: [
        "SPY",      # S&P 500 ETF
        "TLT",      # 20+ Year Treasury Bond ETF
        "GLD",      # Gold ETF
        "BTC-USD"   # Bitcoin
    ])

    # Date range
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"

    # Train/test split
    train_ratio: float = 0.8
    validation_ratio: float = 0.1


# ============================================================================
# AGENT CONFIGURATIONS
# ============================================================================

@dataclass
class DQNConfig:
    """DQN Agent Configuration."""
    name: str = "DQN"
    type: str = "discrete"

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Learning parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Replay buffer
    buffer_capacity: int = 10000
    batch_size: int = 64

    # Target network
    target_update_freq: int = 10  # Update target network every N steps

    # Advanced features
    use_per: bool = False  # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    use_double_dqn: bool = False
    use_dueling: bool = False
    use_noisy: bool = False


@dataclass
class SACConfig:
    """SAC Agent Configuration."""
    name: str = "SAC"
    type: str = "continuous"

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient

    # Temperature (entropy) parameter
    alpha: float = 0.2
    auto_tune_alpha: bool = True

    # Replay buffer
    buffer_capacity: int = 1000000
    batch_size: int = 256

    # Training
    update_freq: int = 1  # Update every N environment steps
    gradient_steps: int = 1  # Gradient steps per update

    # Exploration
    initial_random_steps: int = 1000


@dataclass
class PPOConfig:
    """PPO Agent Configuration."""
    name: str = "PPO"
    type: str = "continuous"

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor

    # PPO-specific
    clip_epsilon: float = 0.2  # PPO clipping parameter
    gae_lambda: float = 0.95  # GAE lambda
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10  # Optimization epochs per rollout


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

@dataclass
class PortfolioEnvConfig:
    """Portfolio Environment Configuration."""
    name: str = "PortfolioEnv"

    # Initial conditions
    initial_balance: float = 100000.0

    # Transaction costs
    transaction_cost: float = 0.001  # 0.1% per trade

    # Action space
    action_type: Literal["discrete", "continuous"] = "continuous"

    # Reward function
    reward_type: Literal["log_utility", "sharpe", "total_return"] = "log_utility"

    # Risk aversion (for utility-based rewards)
    risk_aversion: float = 2.0

    # Constraints
    max_weight: float = 1.0  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    allow_shorting: bool = False

    # Rebalancing
    rebalance_freq: int = 1  # Rebalance every N days (1 = daily)

    # Features included in state
    state_features: List[str] = field(default_factory=lambda: [
        "current_weights",
        "asset_returns",
        "volatility",
        "previous_weights",
        "regime",
        "vix",
        "portfolio_value",
        "cash_weight"
    ])

    # Episode configuration
    max_episode_steps: Optional[int] = None  # None = use full dataset


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class LRScheduleConfig:
    """Learning rate schedule configuration."""
    type: Literal["constant", "linear", "exponential"] = "constant"
    initial: float = 3e-4
    final: float = 1e-5


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 10  # Stop if no improvement for N evaluations
    min_delta: float = 0.01  # Minimum improvement threshold


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    save_best: bool = True
    save_last: bool = True
    metric: str = "eval_return"  # Metric to determine best model


@dataclass
class TrainingConfig:
    """Training Configuration."""
    # Total training steps
    total_timesteps: int = 200000

    # Evaluation
    eval_freq: int = 5000  # Evaluate every N steps
    eval_episodes: int = 5  # Number of episodes per evaluation

    # Logging
    log_freq: int = 1000  # Log metrics every N steps
    save_freq: int = 10000  # Save model every N steps

    # Parallel environments (for on-policy algorithms)
    n_envs: int = 4

    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # Learning rate schedule
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

@dataclass
class OutputConfig:
    """Output directories configuration."""
    results_dir: str = "results"
    logs_dir: str = "logs"
    plots_dir: str = "results/plots"
    tensorboard_dir: str = "runs"


@dataclass
class ModelsConfig:
    """Model storage configuration."""
    save_dir: str = "models"
    checkpoint_freq: int = 10000
    keep_n_checkpoints: int = 5


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    track: bool = True
    platform: Literal["mlflow", "wandb", "tensorboard"] = "mlflow"
    experiment_name: str = "portfolio_allocation"
    run_name: Optional[str] = None  # Auto-generated if None


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    # Project
    project: ProjectConfig = field(default_factory=ProjectConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Agents
    agent_dqn: DQNConfig = field(default_factory=DQNConfig)
    agent_sac: SACConfig = field(default_factory=SACConfig)
    agent_ppo: PPOConfig = field(default_factory=PPOConfig)

    # Environment
    environment: PortfolioEnvConfig = field(default_factory=PortfolioEnvConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output
    output: OutputConfig = field(default_factory=OutputConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # System
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def get_agent_config(self, agent_name: str):
        """Get agent configuration by name."""
        agent_map = {
            "dqn": self.agent_dqn,
            "sac": self.agent_sac,
            "ppo": self.agent_ppo
        }
        return agent_map.get(agent_name.lower())


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_dqn_config() -> Config:
    """Get configuration optimized for DQN training."""
    config = Config()
    config.environment.action_type = "discrete"
    return config


def get_sac_config() -> Config:
    """Get configuration optimized for SAC training."""
    config = Config()
    config.environment.action_type = "continuous"
    return config


def get_ppo_config() -> Config:
    """Get configuration optimized for PPO training."""
    config = Config()
    config.environment.action_type = "continuous"
    config.training.n_envs = 8  # PPO benefits from more parallel envs
    return config


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create and use configuration
    config = get_default_config()

    print(f"Project: {config.project.name} v{config.project.version}")
    print(f"Assets: {config.data.assets}")
    print(f"DQN hidden dims: {config.agent_dqn.hidden_dims}")
    print(f"SAC learning rate: {config.agent_sac.learning_rate}")
    print(f"Training timesteps: {config.training.total_timesteps}")

    # Get specific agent config
    dqn_config = config.get_agent_config("dqn")
    print(f"\nDQN Config: {dqn_config.name}")
    print(f"Epsilon decay: {dqn_config.epsilon_decay}")
