# Interactive Dashboard Guide

**Project:** Deep RL Portfolio Allocation
**Date:** 2025-10-04
**Version:** 2.0

---

## 📊 Available Dashboards

The project includes **three comprehensive interactive dashboards** built with Streamlit and Plotly:

### 1. **Enhanced Dashboard** (Production-Ready)
**File:** `app/enhanced_dashboard.py`
**Purpose:** Main production dashboard with tested components
**Status:** ✅ 16/16 tests passing

### 2. **Analytics Dashboard** (NEW)
**File:** `app/analytics_dashboard.py`
**Purpose:** Advanced portfolio analytics and regime analysis
**Status:** ✅ Production-ready

### 3. **Training Monitor** (NEW)
**File:** `app/training_monitor_dashboard.py`
**Purpose:** Real-time RL agent training monitoring
**Status:** ✅ Production-ready

---

## 🚀 Quick Start

### Launch Analytics Dashboard
```bash
streamlit run app/analytics_dashboard.py
```

### Launch Training Monitor
```bash
streamlit run app/training_monitor_dashboard.py
```

### Launch Enhanced Dashboard (Original)
```bash
streamlit run app/enhanced_dashboard.py
```

**Access:** Open browser to `http://localhost:8501`

---

## 📊 Analytics Dashboard Features

### **Tab 1: Portfolio Overview**

**Key Metrics Display:**
- Total Return
- Sharpe Ratio
- Volatility
- Maximum Drawdown

**Interactive Visualizations:**
- **Normalized Asset Prices:** Compare multiple assets on base-100 scale
- **Returns Distribution:** Histogram with normal distribution overlay
- **Statistical Analysis:** Mean, std dev, skewness, kurtosis

**Features:**
- Multi-asset selection
- Date range filtering
- Interactive hover tooltips
- Zoom and pan functionality

---

### **Tab 2: Regime Analysis**

**Regime Detection:**
- Switch between GMM and HMM methods
- Visual regime distribution (pie chart + bar chart)
- Regime-colored price evolution

**Performance by Regime:**
- Mean return per regime
- Volatility per regime
- Sharpe ratio per regime
- Max drawdown per regime

**Visualizations:**
- Regime distribution pie chart
- Regime frequency bar chart
- Price chart colored by market regime
- Regime statistics table

---

### **Tab 3: Risk Analytics**

**Risk Metrics:**
- Value at Risk (VaR 95%)
- Conditional VaR (CVaR / Expected Shortfall)
- Omega Ratio

**Interactive Charts:**
- **Drawdown Analysis:** Time series of portfolio drawdown
- **Rolling Volatility:** Configurable window (30-252 days)
- **Risk Evolution:** Track risk metrics over time

**Features:**
- Adjustable rolling windows
- Interactive date selection
- Real-time metric calculation

---

### **Tab 4: Strategy Comparison**

**Multi-Strategy Analysis:**
- Portfolio value evolution
- Risk-return scatter plot
- Drawdown comparison
- Statistical significance tests

**Performance Metrics:**
- Side-by-side comparison table
- Percentage improvements
- Crisis period performance

*(Note: Fully functional once models are trained)*

---

### **Tab 5: Training Monitor**

**Training Progress:**
- Episode reward curves (raw + moving averages)
- Episode length tracking
- Real-time updates

**Quick Metrics:**
- Total episodes completed
- Mean reward (last 100 episodes)
- Best reward achieved
- Training speed (episodes/hour)

---

## 🎓 Training Monitor Dashboard

### **Real-Time Monitoring**

**Auto-Refresh:**
- Enable 5-second auto-refresh for live updates
- Monitor training as it happens
- Automatic plot updates

**Key Metrics:**
- Total episodes
- Mean reward (100-episode moving average)
- Best reward achieved
- Mean episode length
- Training speed (ep/hr)

### **Training Curves**

**Episode Rewards:**
- Raw values (transparent line)
- Moving averages (MA-10, MA-50, MA-100)
- Trend analysis

**Episode Lengths:**
- Raw lengths
- 50-episode moving average
- Convergence tracking

### **Statistical Analysis**

**Reward Statistics:**
- Mean, std dev, min, max
- Median, 25th/75th percentiles
- Distribution histogram

**Trend Analysis:**
- Improving vs declining trend
- Stability assessment
- Convergence estimation
- Percentage improvement

### **Data Export**

**Export Options:**
- Download training stats (CSV)
- Download summary report (JSON)
- Timestamped filenames

**Use Cases:**
- Share results with team
- Create publication figures
- Archive training runs

---

## 🎨 Interactive Visualizations Module

### **InteractiveVisualizer Class**

**Location:** `src/visualization/interactive_plots.py`

**Available Plot Types:**

1. **Portfolio Evolution**
   ```python
   viz.plot_portfolio_evolution(strategies, title="...", save_path="...")
   ```
   - Multiple strategies on same chart
   - Interactive legend
   - Hover tooltips with date/value

2. **Returns Distribution**
   ```python
   viz.plot_returns_distribution(strategies, ...)
   ```
   - Overlaid histograms
   - Transparency for comparison
   - Percentage returns

3. **Regime-Colored Prices**
   ```python
   viz.plot_regime_colored_prices(prices, regimes, ...)
   ```
   - Color-coded by market regime
   - Bull/Bear/Volatile periods
   - Visual regime identification

4. **Drawdown Comparison**
   ```python
   viz.plot_drawdown_comparison(strategies, ...)
   ```
   - Multiple strategies
   - Filled area charts
   - Percentage drawdown

5. **Risk-Return Scatter**
   ```python
   viz.plot_risk_return_scatter(metrics, ...)
   ```
   - Strategy positioning
   - Sharpe ratio annotations
   - Optimal frontier visualization

6. **Rolling Metrics**
   ```python
   viz.plot_rolling_metrics(strategies, window=252, metric='sharpe', ...)
   ```
   - Configurable window
   - Multiple metrics (sharpe, volatility, return)
   - Time series evolution

7. **Training Progress**
   ```python
   viz.plot_training_progress(training_stats, metrics=[...], ...)
   ```
   - Multi-metric subplots
   - Raw + moving average
   - Episode tracking

8. **Correlation Matrix**
   ```python
   viz.plot_correlation_matrix(returns, ...)
   ```
   - Heatmap visualization
   - Color-coded correlations
   - Hover values

9. **Regime Transitions**
   ```python
   viz.plot_regime_transitions(transition_matrix, ...)
   ```
   - Probability heatmap
   - Regime flow analysis
   - Transition probabilities

### **Customization Options**

**Themes:**
- `'plotly_dark'` (default)
- `'plotly'`
- `'plotly_white'`
- `'seaborn'`
- `'ggplot2'`

**Color Schemes:**
```python
colors = {
    'bull': '#2ecc71',      # Green
    'bear': '#e74c3c',      # Red
    'volatile': '#f39c12',  # Orange
    'dqn': '#3498db',       # Blue
    'ppo': '#9b59b6',       # Purple
    'merton': '#95a5a6',    # Gray
    'spy': '#1abc9c'        # Teal
}
```

**Save Formats:**
- HTML (interactive)
- PNG (static, via `fig.write_image()`)
- SVG (vector graphics)
- PDF (publication-ready)

---

## 🛠️ Configuration & Customization

### **Dashboard Settings**

**Page Configuration:**
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Theme Selection:**
- Users can select chart theme in sidebar
- Persists across tabs
- Real-time theme switching

**Date Range Filtering:**
- Interactive date picker
- Filters all visualizations
- Min/max date validation

### **Custom CSS Styling**

**Metric Cards:**
```css
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
```

**Status Indicators:**
```css
.positive { color: #2ecc71; }
.negative { color: #e74c3c; }
.status-running { color: #2ecc71; }
```

---

## 📈 Usage Examples

### Example 1: Comparing Strategies

```python
from src.visualization.interactive_plots import InteractiveVisualizer

viz = InteractiveVisualizer(theme='plotly_dark')

strategies = {
    'DQN': dqn_portfolio_values,
    'PPO': ppo_portfolio_values,
    'Merton': merton_portfolio_values
}

fig = viz.plot_portfolio_evolution(
    strategies=strategies,
    title="Strategy Performance Comparison",
    save_path="results/strategy_comparison.html"
)
```

### Example 2: Regime Analysis

```python
fig = viz.plot_regime_colored_prices(
    prices=data['price_SPY'],
    regimes=data['regime_gmm'],
    regime_names={0: 'Bull', 1: 'Bear', 2: 'Volatile'},
    title="SPY Price by Market Regime",
    save_path="results/regime_analysis.html"
)
```

### Example 3: Training Monitoring

```python
# Load training stats
stats = pd.read_csv("models/ppo/training_stats.csv")

# Create training progress plot
fig = viz.plot_training_progress(
    training_stats=stats,
    metrics=['episode_reward', 'episode_length'],
    title="PPO Training Progress"
)
```

### Example 4: Risk Analysis

```python
# Calculate rolling Sharpe ratio
fig = viz.plot_rolling_metrics(
    strategies={'SPY': spy_returns},
    window=252,
    metric='sharpe',
    title="Rolling 1-Year Sharpe Ratio"
)
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Dashboard won't start**
```bash
# Check Streamlit installation
pip install streamlit plotly --upgrade

# Run from project root
cd "d:\Stochastic Control for Continuous - Time Portfolios"
streamlit run app/analytics_dashboard.py
```

**2. Data not loading**
- Check file paths in sidebar
- Ensure CSV files exist
- Verify date format in index

**3. Plots not displaying**
- Refresh browser (Ctrl+F5)
- Check browser console for errors
- Verify Plotly installation: `pip install plotly --upgrade`

**4. Slow performance**
- Reduce date range
- Disable auto-refresh
- Clear Streamlit cache: `streamlit cache clear`

### Performance Optimization

**Caching:**
```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(path):
    return pd.read_csv(path)
```

**Lazy Loading:**
- Load data only when tabs are selected
- Use `st.spinner()` for long operations
- Implement pagination for large datasets

---

## 📊 Dashboard Deployment

### Option 1: Local Deployment
```bash
# Run on specific port
streamlit run app/analytics_dashboard.py --server.port 8502

# Run in headless mode
streamlit run app/analytics_dashboard.py --server.headless true
```

### Option 2: Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/analytics_dashboard.py"]
```

**Build and Run:**
```bash
docker build -t rl-dashboard .
docker run -p 8501:8501 rl-dashboard
```

### Option 3: Streamlit Cloud

1. Push to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Select `app/analytics_dashboard.py`
5. Deploy!

**Public URL:** `https://your-app.streamlit.app`

---

## 📝 Best Practices

### Dashboard Design

1. **Progressive Disclosure:** Show summary first, details on demand
2. **Consistent Layout:** Use columns for aligned metrics
3. **Interactive Controls:** Sliders, dropdowns, checkboxes in sidebar
4. **Clear Labels:** Descriptive titles and axis labels
5. **Status Indicators:** Color-coded metrics (green/red)

### Performance

1. **Cache Aggressively:** Use `@st.cache_data` for data loading
2. **Lazy Computation:** Calculate only when needed
3. **Pagination:** Limit data display for large datasets
4. **Debounce Inputs:** Avoid re-running on every keystroke

### Visualization

1. **Interactive by Default:** Use Plotly for all charts
2. **Consistent Colors:** Use theme colors throughout
3. **Hover Information:** Rich tooltips with context
4. **Export Options:** Allow saving plots as HTML/PNG

---

## 🎯 Future Enhancements

### Planned Features

- [ ] **Multi-Model Comparison:** Compare multiple training runs
- [ ] **A/B Testing:** Statistical comparison of hyperparameters
- [ ] **Alerts System:** Notify when training metrics exceed thresholds
- [ ] **Experiment Tracking:** MLflow integration
- [ ] **Real-Time Streaming:** WebSocket for live updates
- [ ] **Advanced Analytics:** Feature importance, SHAP values
- [ ] **Mobile Responsive:** Optimized layouts for mobile devices

### Integration Possibilities

- **TensorBoard:** Embed TensorBoard in Streamlit
- **Weights & Biases:** W&B experiment tracking
- **Optuna:** Live hyperparameter optimization visualization
- **Ray Tune:** Distributed training monitoring

---

## 📚 Additional Resources

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Python Docs](https://plotly.com/python/)
- [Plotly Dash](https://dash.plotly.com/)

### Examples
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Plotly Examples](https://plotly.com/python/basic-charts/)

### Tutorials
- [Building Data Apps](https://docs.streamlit.io/get-started/tutorials)
- [Interactive Visualizations](https://plotly.com/python/getting-started/)

---

## 🙏 Credits

**Technologies:**
- **Streamlit** - Dashboard framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

---

*🤖 Generated with [Claude Code](https://claude.com/claude-code)*
*Co-Authored-By: Claude <noreply@anthropic.com>*
