# 🚀 Deployment Guide

## Streamlit Cloud Deployment

### Deploying the Agentic Portfolio Manager

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/

2. **Click "New app"**

3. **Enter deployment settings:**
   ```
   Repository: mohin-io/Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset
   Branch: master
   Main file path: app_agentic.py
   App URL (optional): stochastic-control-agentic
   ```

4. **Advanced settings (optional):**
   - Python version: 3.9 or higher
   - Keep default resource settings

5. **Click "Deploy"**

The app will be available at: `https://stochastic-control-agentic.streamlit.app`

---

### Deploying the Classic RL Dashboard

For the original dashboard, use the same steps but:

```
Main file path: app.py
App URL (optional): stochastic-control-for-continuous-time-portfolios
```

The app will be available at: `https://stochastic-control-for-continuous-time-portfolios.streamlit.app`

---

## Local Deployment

### Run Agentic Dashboard Locally

```bash
# Navigate to project directory
cd "d:\Stochastic Control for Continuous - Time Portfolios"

# Install dependencies (if not already installed)
pip install -r requirements.txt
pip install -r requirements-app.txt

# Run the agentic app
streamlit run app_agentic.py
```

The dashboard will open at: `http://localhost:8501`

### Run Classic Dashboard Locally

```bash
streamlit run app.py
```

---

## Requirements

Both dashboards require:
- Python 3.9+
- All dependencies from `requirements.txt` and `requirements-app.txt`
- No trained models required (uses synthetic data for demo)

---

## Troubleshooting

### Issue: ModuleNotFoundError for agents

**Solution:** The app automatically adds `src/` to the Python path. If issues persist:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
```

### Issue: Data files not found

**Solution:** The apps use fallback synthetic data generation if real data is missing. This is intentional for demo purposes.

### Issue: Memory errors on Streamlit Cloud

**Solution:** The free tier has 1GB RAM limit. The apps are optimized to stay within this limit by:
- Using cached data loading
- Limiting historical data to 252 days (1 year)
- Generating synthetic data instead of loading large files

---

## Repository URLs

**Current repo:** `https://github.com/mohin-io/Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset`

**Note:** If the repository was renamed, update the deployment settings accordingly.

---

## Post-Deployment

After deployment:

1. **Test all features:**
   - ✅ Agent Dashboard - Check alert banners and agent cards
   - ✅ Volatility Analysis - Verify charts load
   - ✅ Risk Management - Check VaR/CVaR calculations
   - ✅ Portfolio Allocation - Verify weight recommendations
   - ✅ Forecasting - Check volatility predictions

2. **Update README.md** with actual deployment URLs

3. **Share the app:**
   - The app URL is public and can be shared directly
   - No authentication required for viewing

---

## Production Recommendations

For production use:

1. **Connect to real data sources:**
   - Replace synthetic data with live market data feeds
   - Use APIs: Yahoo Finance, Alpha Vantage, IEX Cloud

2. **Add authentication:**
   - Use Streamlit's authentication features
   - Implement user-specific portfolios

3. **Enable persistent storage:**
   - Store portfolio state in database (PostgreSQL, MongoDB)
   - Save rebalancing history

4. **Set up monitoring:**
   - Track app performance
   - Monitor agent recommendations
   - Log decisions for audit trail

5. **Upgrade resources:**
   - Consider Streamlit Teams/Enterprise for more resources
   - Use dedicated servers for high-frequency trading

---

**Deployed!** 🎉

Both dashboards are now live and accessible to anyone with the URL.
