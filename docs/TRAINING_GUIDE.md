# RL Agent Training Guide

**Date:** October 4, 2025
**Status:** Training Infrastructure Ready
**Progress:** Option 3 In Progress

---

## 🎯 Overview

This guide explains how to train the three RL agents (DQN, PPO, SAC) on the portfolio allocation task.

---

## 📋 Training Scripts Available

### 1. DQN Training
**Script:** `scripts/train_dqn.py`

```bash
python scripts/train_dqn.py \
  --data data/processed/dataset_with_regimes.csv \
  --episodes 1000 \
  --save models/dqn_final.pth \
  --device cuda  # or cpu
```

**Parameters:**
- `--data`: Path to processed dataset
- `--episodes`: Number of training episodes (default: 1000)
- `--save`: Path to save trained model
- `--device`: Training device (cpu or cuda)

**Expected Training Time:** ~2-3 hours for 1000 episodes on CPU

---

### 2. PPO Training
**Script:** `scripts/train_ppo_optimized.py`

```bash
python scripts/train_ppo_optimized.py \
  --n-envs 8 \
  --total-timesteps 500000 \
  --learning-rate 3e-4 \
  --output-dir models/ppo
```

**Parameters:**
- `--n-envs`: Number of parallel environments (default: 8)
- `--total-timesteps`: Total training timesteps (default: 500000)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--output-dir`: Output directory for models and logs

**Expected Training Time:** ~4-6 hours for 500K timesteps with 8 parallel envs

---

### 3. SAC Training
**Script:** `scripts/train_sac.py`

```bash
python scripts/train_sac.py \
  --data data/processed/complete_dataset.csv \
  --total-timesteps 500000 \
  --save models/sac_final.pth
```

**Parameters:**
- `--data`: Path to processed dataset
- `--total-timesteps`: Total training timesteps (default: 200000)
- `--save`: Path to save trained model

**Expected Training Time:** ~3-5 hours for 500K timesteps on CPU

---

## 🚀 Quick Start - Train All Agents

### Sequential Training (Recommended)

```bash
# 1. Train DQN
python scripts/train_dqn.py --episodes 1000 --save models/dqn_trained.pth

# 2. Train PPO
python scripts/train_ppo_optimized.py --total-timesteps 500000 --output-dir models/ppo

# 3. Train SAC
python scripts/train_sac.py --total-timesteps 500000 --save models/sac_trained.pth
```

**Total Time:** ~9-14 hours

---

## 📊 Monitoring Training

### During Training

All scripts output progress with `tqdm` progress bars and periodic metrics:

```
Training: 100%|██████| 1000/1000 [2:15:32<00:00,  8.13s/it]
Episode 100: Avg Reward = 5.23, Portfolio Value = $125,430
Episode 200: Avg Reward = 8.15, Portfolio Value = $142,891
...
```

### After Training

Check trained models:
```bash
ls -lh models/
# dqn_trained.pth
# ppo/ppo_final.pth
# sac_trained.pth
```

---

## 🔧 Training Configuration

### Environment Setup
```python
env = PortfolioEnv(
    data=train_data,
    initial_balance=100000.0,
    transaction_cost=0.001,  # 0.1%
    action_type='continuous',  # or 'discrete' for DQN
    reward_type='log_utility'
)
```

### Data Split
- **Training:** 80% of data (~2,056 timesteps)
- **Testing:** 20% of data (~514 timesteps)
- **Date Range:** 2014-2024

---

## 📈 Expected Results

Based on initial experiments and similar implementations:

### DQN (1000 episodes)
- Final Portfolio Value: $120K - $150K
- Sharpe Ratio: 0.4 - 0.8
- Training Episodes: 1000
- Convergence: ~500-700 episodes

### PPO (500K timesteps)
- Final Portfolio Value: $130K - $180K
- Sharpe Ratio: 0.6 - 1.2
- Training Episodes: ~800-1200
- Convergence: ~300K timesteps

### SAC (500K timesteps)
- Final Portfolio Value: $140K - $200K
- Sharpe Ratio: 0.7 - 1.4
- Training Episodes: ~800-1200
- Convergence: ~250K timesteps

**Note:** Actual results depend on hyperparameters, random seed, and market data.

---

## 🎓 Hyperparameter Tuning

### DQN
```python
learning_rate = 1e-4
gamma = 0.99
epsilon_decay = 0.995
buffer_capacity = 10000
batch_size = 64
```

### PPO
```python
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
n_epochs = 10
batch_size = 64
```

### SAC
```python
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2  # or auto-tuned
buffer_capacity = 100000
batch_size = 256
```

---

## 📝 Saved Outputs

### Model Checkpoints
- `models/dqn_trained.pth` - Trained DQN weights
- `models/ppo/ppo_final.pth` - Trained PPO weights
- `models/sac_trained.pth` - Trained SAC weights

### Training Logs
- Episode rewards
- Portfolio values
- Sharpe ratios
- Training losses

---

## 🔍 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `buffer_capacity`
- Use `--device cpu` instead of cuda

### Slow Training
- Reduce `--episodes` or `--total-timesteps`
- Use fewer parallel environments (`--n-envs`)
- Use smaller network (`hidden_dims`)

### Not Converging
- Increase learning rate
- Adjust epsilon decay (DQN)
- Increase training timesteps
- Check data quality

---

## 🎯 Next Steps After Training

1. **Evaluate Agents:**
   - Test on held-out data
   - Calculate performance metrics
   - Compare vs baselines

2. **Create Adapters:**
   - Integrate with BacktestEngine
   - Run comprehensive backtests
   - Generate comparison reports

3. **Analysis:**
   - Training curve visualization
   - Portfolio allocation analysis
   - Crisis period performance

---

## 📊 Integration with Backtesting

Once trained, integrate with BacktestEngine:

```python
from src.backtesting import BacktestEngine, BacktestConfig
from src.agents import DQNAgent

# Load trained agent
agent = DQNAgent.load('models/dqn_trained.pth')

# Create adapter
class DQNAdapter(Strategy):
    def __init__(self, agent):
        self.agent = agent

    def allocate(self, data, current_idx, current_weights):
        state = self._prepare_state(data, current_idx)
        action = self.agent.select_action(state, epsilon=0)  # No exploration
        return self._action_to_weights(action)

# Run backtest
engine = BacktestEngine(BacktestConfig())
results = engine.run(DQNAdapter(agent), data, returns)
```

---

## 🚧 Current Status

**Infrastructure:** ✅ Complete
- [x] Training scripts ready
- [x] Environments configured
- [x] Agents implemented
- [x] Data prepared

**Training:** 🔄 In Progress (Option 3)
- [ ] DQN training (1000 episodes)
- [ ] PPO training (500K timesteps)
- [ ] SAC training (500K timesteps)

**Next:** After training completion
- Create RL agent adapters for BacktestEngine
- Run comprehensive evaluation
- Generate comparison reports

---

*🤖 Guide created with [Claude Code](https://claude.com/claude-code)*
*Date: October 4, 2025*
