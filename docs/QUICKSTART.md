# Quick Start Guide

Get up and running with Deep RL Portfolio Allocation in 5 minutes.

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Clone Repository
```bash
git clone https://github.com/mohin-io/deep-rl-portfolio-allocation.git
cd deep-rl-portfolio-allocation
```

### Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Download Market Data

```bash
python src/data_pipeline/download.py
```

This downloads:
- **Assets:** SPY, TLT, GLD, BTC-USD (2010-2025)
- **VIX:** Volatility index
- **Treasury rates:** 10-year yields

Data saved to: `data/raw/`

### Step 2: Preprocess Data

```python
from src.data_pipeline.preprocessing import DataPreprocessor
import pandas as pd

# Load raw data
prices = pd.read_csv("data/raw/asset_prices_1d.csv", index_col=0, parse_dates=True)
vix = pd.read_csv("data/raw/vix.csv", index_col=0, parse_dates=True).squeeze()
treasury = pd.read_csv("data/raw/treasury_10y.csv", index_col=0, parse_dates=True).squeeze()

# Prepare dataset
preprocessor = DataPreprocessor()
dataset = preprocessor.prepare_dataset(prices, vix, treasury)
```

Dataset saved to: `data/processed/complete_dataset.csv`

### Step 3: Train Regime Detection

```python
from src.regime_detection.gmm_classifier import GMMRegimeDetector

# Load processed data
returns = dataset[[col for col in dataset.columns if col.startswith('return_')]]

# Train GMM
detector = GMMRegimeDetector(n_regimes=3)
detector.fit(returns, dataset['VIX'])
detector.save('models/gmm_regime_detector.pkl')

# Get regime statistics
stats = detector.get_regime_statistics(returns, dataset['VIX'])
print(stats)
```

### Step 4: Train DQN Agent

```bash
python scripts/train_dqn.py --episodes 1000 --device cpu
```

Options:
- `--data`: Path to dataset (default: `data/processed/complete_dataset.csv`)
- `--episodes`: Number of training episodes (default: 1000)
- `--save`: Model save path (default: `models/dqn_agent.pth`)
- `--device`: `cpu` or `cuda` (default: `cpu`)

Training output:
```
Episode 50/1000
  Avg Reward (50 eps): 0.234
  Avg Length: 250.5
  Epsilon: 0.780
  Final Portfolio Value: $105,234.56
  Sharpe Ratio: 1.234
```

### Step 5: Evaluate Performance

```python
from src.environments.portfolio_env import PortfolioEnv
from src.agents.dqn_agent import DQNAgent
import pandas as pd

# Load test data
test_data = pd.read_csv("data/processed/complete_dataset.csv", index_col=0, parse_dates=True)
test_data = test_data.iloc[-500:]  # Last 500 days

# Create environment
env = PortfolioEnv(data=test_data, action_type='discrete')

# Load trained agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.load('models/dqn_agent.pth')

# Run evaluation
state, _ = env.reset()
done = False

while not done:
    action = agent.select_action(state, epsilon=0.0)  # Greedy
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Get metrics
metrics = env.get_portfolio_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

---

## 🎯 Common Tasks

### Run Merton Baseline

```python
from src.baselines.merton_strategy import MertonStrategy
import pandas as pd

# Load returns
returns = dataset[[col for col in dataset.columns if col.startswith('return_')]]

# Run backtest
strategy = MertonStrategy(risk_free_rate=0.02)
results = strategy.backtest(returns, initial_value=100000.0)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Compare DQN vs Merton

```python
import matplotlib.pyplot as plt

# Assuming you have both results
plt.figure(figsize=(12, 6))
plt.plot(dqn_results['portfolio_values'], label='DQN', linewidth=2)
plt.plot(merton_results['portfolio_values'], label='Merton', linewidth=2)
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.title('DQN vs Merton Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('docs/figures/performance/dqn_vs_merton.png', dpi=300)
plt.show()
```

---

## 🔧 Configuration

### Environment Configuration

Customize the portfolio environment:

```python
env = PortfolioEnv(
    data=dataset,
    initial_balance=100000.0,      # Starting capital
    transaction_cost=0.001,         # 0.1% per trade
    risk_free_rate=0.02,           # 2% annual
    window_size=60,                # Look-back window
    action_type='continuous',      # or 'discrete'
    reward_type='log_utility'      # or 'sharpe', 'return'
)
```

### DQN Hyperparameters

```python
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-4,           # Adam learning rate
    gamma=0.99,                   # Discount factor
    epsilon_start=1.0,            # Initial exploration
    epsilon_end=0.01,             # Final exploration
    epsilon_decay=0.995,          # Decay rate
    buffer_capacity=10000,        # Replay buffer size
    batch_size=64,                # Training batch size
    target_update_freq=10         # Target network update
)
```

---

## 📊 Expected Outputs

After completing the quick start, you should have:

### Files Generated
```
data/
├── raw/
│   ├── asset_prices_1d.csv
│   ├── vix.csv
│   └── treasury_10y.csv
├── processed/
│   └── complete_dataset.csv
└── regime_labels/
    └── gmm_regimes.csv

models/
├── gmm_regime_detector.pkl
├── hmm_regime_detector.pkl
└── dqn_agent.pth
```

### Performance Metrics
- Total Return: 15-25% (expected range)
- Sharpe Ratio: 0.8-1.5
- Max Drawdown: 10-20%
- Training Time: 30-60 minutes (CPU), 10-15 minutes (GPU)

---

## 🐛 Troubleshooting

### Issue: yfinance download fails
**Solution:** Check internet connection, try again, or use smaller date range

### Issue: FRED API not working
**Solution:** Set `FRED_API_KEY` environment variable or use mock data

```bash
export FRED_API_KEY="your_api_key_here"  # Get from https://fred.stlouisfed.org/docs/api/api_key.html
```

### Issue: Out of memory during training
**Solution:** Reduce batch size or buffer capacity

```python
agent = DQNAgent(..., buffer_capacity=5000, batch_size=32)
```

### Issue: Training not converging
**Solution:** Try different hyperparameters or increase episodes

```bash
python scripts/train_dqn.py --episodes 2000
```

---

## 📚 Next Steps

After completing the quick start:

1. **Experiment with Hyperparameters**
   - Try different learning rates
   - Adjust exploration parameters
   - Test various reward functions

2. **Implement PPO Agent**
   - Create `src/agents/ppo_agent.py`
   - Compare continuous vs discrete actions

3. **Add More Visualizations**
   - Regime-colored price charts
   - Allocation heatmaps
   - Drawdown comparisons

4. **Build Dashboard**
   - Create Streamlit app
   - Interactive strategy comparison
   - Real-time recommendations

5. **Deploy API**
   - FastAPI endpoint
   - Docker containerization
   - Cloud deployment

---

## 💡 Tips

- **Start Small:** Begin with 100-200 episodes to test setup
- **Monitor Progress:** Check logs every 50 episodes
- **Save Checkpoints:** Training can take time, save intermediate models
- **Use GPU:** If available, speeds up training 3-5x
- **Experiment:** Try different asset combinations, regimes, rewards

---

## 🤝 Need Help?

- **Documentation:** See `docs/PLAN.md` for detailed architecture
- **Code Examples:** Each module has test/example code at the bottom
- **Issues:** Report on GitHub: https://github.com/mohin-io/deep-rl-portfolio-allocation/issues

---

**Happy Trading with RL! 📈🤖**
