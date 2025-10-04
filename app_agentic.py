"""
Agentic Portfolio Management Dashboard
Multi-Agent System for Volatility-Aware Portfolio Allocation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agents.volatility_agents import (
    AgentCoordinator,
    VolatilityRegime,
    MarketRegime
)

# Page config
st.set_page_config(
    page_title="Agentic Portfolio Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-top: 0;
    }
    .alert-red {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
    }
    .alert-yellow {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
    }
    .alert-green {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
    }
    .agent-card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Agentic Portfolio Manager</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Agent System for Volatility-Aware Asset Allocation</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 System Overview")
    st.markdown("""
    **Active Agents:**
    - 🔍 Volatility Detection Agent
    - ⚠️ Risk Management Agent
    - 📊 Regime Detection Agent
    - ⚖️ Adaptive Rebalancing Agent
    - 📈 Volatility Forecasting Agent
    - 🎛️ Agent Coordinator
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=1000000,
        step=50000
    )

    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
        value="Moderate"
    )

    rebalancing_mode = st.selectbox(
        "Rebalancing Mode",
        ["Adaptive (Agent-Controlled)", "Daily", "Weekly", "Monthly"]
    )

    st.markdown("---")
    st.markdown("### 📊 Agent Status")

# Load data
@st.cache_data
def load_market_data():
    """Load historical market data"""
    data_path = Path("data/processed/complete_dataset.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df
    else:
        # Generate synthetic data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)

        spy = 300 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))
        tlt = 120 * np.exp(np.cumsum(np.random.normal(0.0002, 0.008, len(dates))))
        gld = 150 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        btc = 8000 * np.exp(np.cumsum(np.random.normal(0.002, 0.04, len(dates))))

        df = pd.DataFrame({
            'price_SPY': spy,
            'price_TLT': tlt,
            'price_GLD': gld,
            'price_BTC': btc
        }, index=dates)

        for col in df.columns:
            df[f'return_{col.split("_")[1]}'] = df[col].pct_change()

        df['vix'] = 16 + np.random.normal(0, 5, len(dates)) + \
                    np.where(np.random.random(len(dates)) > 0.95, 20, 0)
        df['vix'] = df['vix'].clip(10, 80)

        return df.dropna()

# Initialize agent coordinator
@st.cache_resource
def get_agent_coordinator():
    return AgentCoordinator()

df = load_market_data()
coordinator = get_agent_coordinator()

# Extract assets
price_cols = [col for col in df.columns if col.startswith('price_')]
return_cols = [col for col in df.columns if col.startswith('return_')]
n_assets = len(price_cols)

# Current portfolio state
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = initial_capital
    st.session_state.weights = np.ones(n_assets) / n_assets
    st.session_state.days_since_rebalance = 0
    st.session_state.rebalance_history = []

# Get latest data
latest_prices = df[price_cols].iloc[-1]
latest_returns = df[return_cols].iloc[-252:]  # Last year
latest_vix = df['vix'].iloc[-1] if 'vix' in df.columns else None

# Get agent recommendation
recommendation = coordinator.get_portfolio_recommendation(
    prices=df[price_cols].iloc[-252:],
    returns=df[return_cols].iloc[-252:],
    current_weights=st.session_state.weights,
    portfolio_value=st.session_state.portfolio_value,
    vix=latest_vix,
    days_since_rebalance=st.session_state.days_since_rebalance
)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Agent Dashboard",
    "📊 Volatility Analysis",
    "⚠️ Risk Management",
    "📈 Portfolio Allocation",
    "🔮 Forecasting"
])

# Tab 1: Agent Dashboard
with tab1:
    st.markdown("## 🎛️ Multi-Agent Control Center")

    # Alert banner
    alert_level = recommendation['alert_level']
    vol_signal = recommendation['volatility_signal']

    if alert_level == 'red':
        st.markdown(f"""
        <div class="alert-red">
            <h3>🚨 HIGH VOLATILITY ALERT</h3>
            <p><strong>Current Regime:</strong> {vol_signal.regime.name}</p>
            <p><strong>Volatility:</strong> {vol_signal.current_vol:.2f}% (Forecasted: {vol_signal.forecasted_vol:.2f}%)</p>
            <p><strong>Recommended Action:</strong> Reduce risk exposure, increase cash to {recommendation['recommended_cash']*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    elif alert_level == 'yellow':
        st.markdown(f"""
        <div class="alert-yellow">
            <h3>⚠️ ELEVATED VOLATILITY WARNING</h3>
            <p><strong>Current Regime:</strong> {vol_signal.regime.name}</p>
            <p><strong>Volatility:</strong> {vol_signal.current_vol:.2f}% (Forecasted: {vol_signal.forecasted_vol:.2f}%)</p>
            <p><strong>Recommended Action:</strong> Monitor closely, consider defensive positioning</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-green">
            <h3>✅ NORMAL MARKET CONDITIONS</h3>
            <p><strong>Current Regime:</strong> {vol_signal.regime.name}</p>
            <p><strong>Volatility:</strong> {vol_signal.current_vol:.2f}% (Forecasted: {vol_signal.forecasted_vol:.2f}%)</p>
            <p><strong>Status:</strong> All systems nominal</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Agent status cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 Volatility Detection Agent</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Current Regime", vol_signal.regime.name)
        st.metric("Confidence", f"{vol_signal.confidence*100:.1f}%")
        st.metric("Alert Level", alert_level.upper())

    with col2:
        st.markdown("""
        <div class="agent-card">
            <h4>⚠️ Risk Management Agent</h4>
        </div>
        """, unsafe_allow_html=True)
        risk_signal = recommendation['risk_signal']
        st.metric("Recommended Cash", f"{risk_signal.recommended_cash*100:.1f}%")
        st.metric("VaR Breach", "⚠️ YES" if risk_signal.var_breach else "✅ NO")
        st.metric("Correlation Spike", "⚠️ YES" if risk_signal.correlation_spike else "✅ NO")

    with col3:
        st.markdown("""
        <div class="agent-card">
            <h4>📊 Regime Detection Agent</h4>
        </div>
        """, unsafe_allow_html=True)
        market_regime = recommendation['market_regime']
        st.metric("Market Regime", market_regime.name.replace('_', ' '))
        st.metric("Rebalance Needed", "⚖️ YES" if recommendation['should_rebalance'] else "✅ NO")
        st.metric("Days Since Rebalance", st.session_state.days_since_rebalance)

    st.markdown("---")

    # Portfolio metrics
    st.markdown("### 📊 Current Portfolio Status")

    col1, col2, col3, col4 = st.columns(4)

    portfolio_returns = (df[return_cols].iloc[-1:] * st.session_state.weights).sum().values
    total_return = ((st.session_state.portfolio_value - initial_capital) / initial_capital) * 100

    with col1:
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.0f}")

    with col2:
        st.metric("Total Return", f"{total_return:.2f}%")

    with col3:
        current_vol = (df[return_cols].iloc[-252:] @ st.session_state.weights).std() * np.sqrt(252) * 100
        st.metric("Portfolio Volatility", f"{current_vol:.2f}%")

    with col4:
        sharpe = (total_return - 2) / current_vol if current_vol > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")

    # Weight comparison
    st.markdown("### ⚖️ Portfolio Weights: Current vs Recommended")

    assets = [col.replace('price_', '') for col in price_cols]
    weight_comparison = pd.DataFrame({
        'Asset': assets,
        'Current Weight (%)': st.session_state.weights * 100,
        'Recommended Weight (%)': recommendation['recommended_weights'] * 100,
        'Drift (%)': (st.session_state.weights - recommendation['recommended_weights']) * 100
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Current',
        x=assets,
        y=st.session_state.weights * 100,
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Recommended',
        x=assets,
        y=recommendation['recommended_weights'] * 100,
        marker_color='darkblue'
    ))
    fig.update_layout(
        title="Weight Comparison",
        barmode='group',
        yaxis_title="Weight (%)",
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    st.dataframe(weight_comparison, width='stretch')

    # Rebalance button
    if recommendation['should_rebalance']:
        if st.button("🔄 Execute Rebalance (Agent Recommended)", type="primary"):
            st.session_state.weights = recommendation['recommended_weights']
            st.session_state.days_since_rebalance = 0
            st.session_state.rebalance_history.append({
                'date': datetime.now(),
                'regime': vol_signal.regime.name,
                'weights': recommendation['recommended_weights'].copy()
            })
            st.success("✅ Portfolio rebalanced successfully!")
            st.rerun()


# Tab 2: Volatility Analysis
with tab2:
    st.markdown("## 📊 Comprehensive Volatility Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Volatility Regimes Over Time")

        # Calculate rolling volatility
        portfolio_returns = (df[return_cols] @ st.session_state.weights)
        rolling_vol = portfolio_returns.rolling(20).std() * np.sqrt(252) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-252:],
            y=rolling_vol.iloc[-252:],
            mode='lines',
            name='Realized Vol',
            fill='tozeroy',
            line=dict(color='blue')
        ))

        # Add regime thresholds
        fig.add_hline(y=12, line_dash="dash", line_color="green", annotation_text="Low Vol")
        fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Elevated Vol")
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="High Vol")

        fig.update_layout(
            title="Rolling Volatility (20-day)",
            yaxis_title="Volatility (%)",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("### 🎯 Volatility Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=rolling_vol.dropna(),
            nbinsx=30,
            name='Volatility Distribution',
            marker_color='skyblue'
        ))

        fig.add_vline(
            x=vol_signal.current_vol,
            line_dash="dash",
            line_color="red",
            annotation_text="Current"
        )

        fig.update_layout(
            title="Historical Volatility Distribution",
            xaxis_title="Volatility (%)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # VIX analysis
    if 'vix' in df.columns:
        st.markdown("### 📊 VIX Fear Index Analysis")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-252:],
            y=df['vix'].iloc[-252:],
            mode='lines',
            name='VIX',
            line=dict(color='purple')
        ))

        # Color zones
        fig.add_hrect(y0=0, y1=12, fillcolor="green", opacity=0.1, annotation_text="Calm")
        fig.add_hrect(y0=12, y1=20, fillcolor="yellow", opacity=0.1, annotation_text="Normal")
        fig.add_hrect(y0=20, y1=30, fillcolor="orange", opacity=0.1, annotation_text="Elevated")
        fig.add_hrect(y0=30, y1=100, fillcolor="red", opacity=0.1, annotation_text="High Fear")

        fig.update_layout(
            title="VIX Index (Market Fear Gauge)",
            yaxis_title="VIX Level",
            height=500
        )
        st.plotly_chart(fig, width='stretch')


# Tab 3: Risk Management
with tab3:
    st.markdown("## ⚠️ Advanced Risk Management")

    risk_signal = recommendation['risk_signal']

    col1, col2, col3 = st.columns(3)

    # Calculate risk metrics
    portfolio_returns = (df[return_cols].iloc[-252:] @ st.session_state.weights)

    with col1:
        var_95 = np.percentile(portfolio_returns, 5) * 100
        st.metric("VaR (95%)", f"{var_95:.2f}%", help="Maximum loss expected 95% of the time")

    with col2:
        var_99 = np.percentile(portfolio_returns, 1) * 100
        st.metric("VaR (99%)", f"{var_99:.2f}%", help="Maximum loss expected 99% of the time")

    with col3:
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
        st.metric("CVaR (95%)", f"{cvar_95:.2f}%", help="Expected loss when VaR is breached")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Drawdown Analysis")

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-252:],
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))

        fig.add_hline(
            y=-20,
            line_dash="dash",
            line_color="darkred",
            annotation_text="Max DD Threshold"
        )

        fig.update_layout(
            title="Portfolio Drawdown",
            yaxis_title="Drawdown (%)",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("### 🔗 Asset Correlation Matrix")

        corr_matrix = df[return_cols].iloc[-252:].corr()
        asset_names = [col.replace('return_', '') for col in return_cols]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=asset_names,
            y=asset_names,
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))

        fig.update_layout(
            title="Asset Correlation Matrix",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    # Risk alerts
    st.markdown("### 🚨 Active Risk Alerts")

    alerts = []
    if risk_signal.max_drawdown_alert:
        alerts.append("⚠️ Maximum drawdown threshold exceeded")
    if risk_signal.var_breach:
        alerts.append("⚠️ VaR breach detected")
    if risk_signal.correlation_spike:
        alerts.append("⚠️ Correlation spike - systemic risk warning")

    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("✅ No active risk alerts")


# Tab 4: Portfolio Allocation
with tab4:
    st.markdown("## 📈 Dynamic Portfolio Allocation")

    # Allocation evolution
    st.markdown("### 📊 Recommended Allocation Evolution")

    # Simulate allocation over time (last 60 days)
    allocation_history = []
    for i in range(-60, 0):
        try:
            rec = coordinator.get_portfolio_recommendation(
                prices=df[price_cols].iloc[i-252:i],
                returns=df[return_cols].iloc[i-252:i],
                current_weights=st.session_state.weights,
                portfolio_value=st.session_state.portfolio_value,
                vix=df['vix'].iloc[i] if 'vix' in df.columns else None,
                days_since_rebalance=0
            )
            allocation_history.append(rec['recommended_weights'])
        except:
            continue

    if allocation_history:
        allocation_df = pd.DataFrame(
            allocation_history,
            columns=[col.replace('price_', '') for col in price_cols]
        )

        fig = go.Figure()
        for col in allocation_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(allocation_df))),
                y=allocation_df[col] * 100,
                mode='lines',
                name=col,
                stackgroup='one'
            ))

        fig.update_layout(
            title="Agent-Recommended Allocation Over Time",
            yaxis_title="Allocation (%)",
            xaxis_title="Days Ago",
            height=500
        )
        st.plotly_chart(fig, width='stretch')


# Tab 5: Forecasting
with tab5:
    st.markdown("## 🔮 Volatility Forecasting")

    vol_forecast = recommendation['volatility_forecast']

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 5-Day Volatility Forecast")

        fig = go.Figure()

        # Historical
        historical_vol = (portfolio_returns.rolling(20).std() * np.sqrt(252) * 100).iloc[-30:]
        fig.add_trace(go.Scatter(
            x=list(range(-len(historical_vol), 0)),
            y=historical_vol,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=vol_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Volatility: Historical vs Forecast",
            xaxis_title="Days",
            yaxis_title="Volatility (%)",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("### 📊 Forecast Summary")
        st.metric("Current Vol", f"{vol_signal.current_vol:.2f}%")
        st.metric("Next Day Forecast", f"{vol_forecast[0]:.2f}%")
        st.metric("5-Day Avg Forecast", f"{vol_forecast.mean():.2f}%")

        change = ((vol_forecast[0] - vol_signal.current_vol) / vol_signal.current_vol) * 100
        st.metric("Expected Change", f"{change:+.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🤖 Powered by Multi-Agent Reinforcement Learning System<br>
    Agents: Volatility Detection | Risk Management | Regime Detection | Adaptive Rebalancing | Forecasting
</div>
""", unsafe_allow_html=True)
