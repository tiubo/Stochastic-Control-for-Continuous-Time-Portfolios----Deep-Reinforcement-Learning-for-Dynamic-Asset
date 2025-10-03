"""
Streamlit Dashboard for Deep RL Portfolio Allocation
Interactive visualization and analysis dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Deep RL Portfolio Allocation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 Deep RL for Dynamic Asset Allocation")
st.markdown("**Modern Evolution of Merton's Portfolio Theory**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("### Data Selection")

# Load data
@st.cache_data
def load_data():
    """Load processed dataset."""
    try:
        data = pd.read_csv(
            "data/processed/dataset_with_regimes.csv",
            index_col=0,
            parse_dates=True
        )
        return data
    except FileNotFoundError:
        return None

data = load_data()

if data is None:
    st.error("❌ Data not found! Please run preprocessing scripts first.")
    st.code("python scripts/simple_preprocess.py\npython scripts/train_regime_models.py")
    st.stop()

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(data.index.min(), data.index.max()),
    min_value=data.index.min(),
    max_value=data.index.max()
)

# Filter data
if len(date_range) == 2:
    mask = (data.index >= pd.Timestamp(date_range[0])) & (data.index <= pd.Timestamp(date_range[1]))
    filtered_data = data[mask]
else:
    filtered_data = data

# Regime selector
regime_type = st.sidebar.selectbox(
    "Regime Detection Model",
    ["GMM (Gaussian Mixture)", "HMM (Hidden Markov)"]
)

regime_col = 'regime_gmm' if 'GMM' in regime_type else 'regime_hmm'

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Regime Analysis", "📈 Asset Performance", "ℹ️ About"])

with tab1:
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Days", len(filtered_data))
    with col2:
        st.metric("Assets", 4)
    with col3:
        st.metric("Features", filtered_data.shape[1])
    with col4:
        st.metric("Date Range", f"{(data.index.max() - data.index.min()).days} days")

    st.markdown("### Asset Prices Over Time")

    # Price chart
    price_cols = [col for col in filtered_data.columns if col.startswith('price_')]

    fig = go.Figure()

    for col in price_cols:
        asset_name = col.replace('price_', '')
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data[col],
            name=asset_name,
            mode='lines'
        ))

    fig.update_layout(
        title="Asset Price Trajectories",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Returns distribution
    st.markdown("### Return Distributions")

    return_cols = [col for col in filtered_data.columns if col.startswith('return_')]

    fig2 = go.Figure()

    for col in return_cols:
        asset_name = col.replace('return_', '')
        fig2.add_trace(go.Histogram(
            x=filtered_data[col] * 100,
            name=asset_name,
            opacity=0.7,
            nbinsx=50
        ))

    fig2.update_layout(
        title="Daily Return Distributions (%)",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("🎯 Market Regime Analysis")

    st.markdown(f"### Regime Classification ({regime_type})")

    # Regime distribution
    regime_counts = filtered_data[regime_col].value_counts().sort_index()
    regime_names = {0: 'Bull' if 'gmm' in regime_col else 'Volatile',
                   1: 'Bear' if 'gmm' in regime_col else 'Bull',
                   2: 'Volatile' if 'gmm' in regime_col else 'Bear'}

    regime_df = pd.DataFrame({
        'Regime': [regime_names.get(i, f'Regime {i}') for i in regime_counts.index],
        'Count': regime_counts.values,
        'Percentage': regime_counts.values / regime_counts.sum() * 100
    })

    col1, col2 = st.columns(2)

    with col1:
        # Regime pie chart
        fig3 = px.pie(
            regime_df,
            values='Count',
            names='Regime',
            title='Regime Distribution',
            color='Regime',
            color_discrete_map={'Bull': 'green', 'Bear': 'red', 'Volatile': 'orange'}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Regime stats table
        st.markdown("#### Regime Statistics")
        st.dataframe(regime_df.style.format({'Percentage': '{:.1f}%'}), use_container_width=True)

    # Regime-colored SPY prices
    st.markdown("### SPY Prices Colored by Market Regime")

    regime_colors_map = {
        'Bull': 'green',
        'Bear': 'red',
        'Volatile': 'orange'
    }

    fig4 = go.Figure()

    for regime_id in sorted(filtered_data[regime_col].unique()):
        mask = filtered_data[regime_col] == regime_id
        regime_name = regime_names.get(regime_id, f'Regime {regime_id}')

        fig4.add_trace(go.Scatter(
            x=filtered_data.index[mask],
            y=filtered_data['price_SPY'][mask],
            name=regime_name,
            mode='markers',
            marker=dict(
                size=4,
                color=regime_colors_map.get(regime_name, 'gray'),
                opacity=0.6
            )
        ))

    fig4.update_layout(
        title=f"SPY Price with {regime_type} Regime Classification",
        xaxis_title="Date",
        yaxis_title="SPY Price ($)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.header("📈 Asset Performance Metrics")

    # Calculate metrics for each asset
    st.markdown("### Performance Summary")

    return_cols = [col for col in filtered_data.columns if col.startswith('return_')]
    vol_cols = [col for col in filtered_data.columns if col.startswith('volatility_')]

    metrics_data = []

    for i, ret_col in enumerate(return_cols):
        asset_name = ret_col.replace('return_', '')
        vol_col = f'volatility_{asset_name}'

        returns = filtered_data[ret_col]
        volatility = filtered_data[vol_col].mean() if vol_col in filtered_data.columns else returns.std() * np.sqrt(252)

        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        metrics_data.append({
            'Asset': asset_name,
            'Total Return': f'{total_return*100:.2f}%',
            'Annual Return': f'{annual_return*100:.2f}%',
            'Volatility': f'{volatility*100:.2f}%',
            'Sharpe Ratio': f'{sharpe:.3f}'
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Correlation matrix
    st.markdown("### Asset Return Correlations")

    returns_df = filtered_data[return_cols].copy()
    returns_df.columns = [col.replace('return_', '') for col in returns_df.columns]
    corr_matrix = returns_df.corr()

    fig5 = px.imshow(
        corr_matrix,
        text_auto='.3f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )

    fig5.update_layout(
        title="Asset Return Correlation Matrix",
        height=400
    )

    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### Deep Reinforcement Learning for Dynamic Asset Allocation

    This project represents a **modern evolution of Merton's classical portfolio theory**,
    combining deep reinforcement learning with market regime detection for adaptive asset allocation.

    #### 🎯 Key Features

    - **Real Market Data**: 15 years of daily data (SPY, TLT, GLD, BTC)
    - **Regime Detection**: GMM and HMM models to identify Bull/Bear/Volatile markets
    - **Deep RL**: DQN and PPO agents trained to optimize portfolio allocation
    - **Regime-Aware States**: RL agents receive market regime as part of state space
    - **Transaction Costs**: Realistic modeling of trading costs and slippage

    #### 🏗️ Technical Stack

    - **Data**: Yahoo Finance, FRED API
    - **ML/RL**: PyTorch, Stable-Baselines3, scikit-learn
    - **Visualization**: Streamlit, Plotly, Matplotlib
    - **Environment**: OpenAI Gymnasium

    #### 📊 MDP Formulation

    **State**: Portfolio weights, returns, volatility, regime, macro indicators (34-dim)
    **Action**: Discrete (Decrease/Hold/Increase) or Continuous allocation
    **Reward**: Log utility with transaction cost penalty

    #### 🎓 Methodology

    1. **Data Preprocessing**: Returns, volatility, feature engineering
    2. **Regime Detection**: Unsupervised learning (GMM/HMM) for market states
    3. **RL Training**: DQN/PPO agents learn optimal allocation policies
    4. **Backtesting**: Compare vs Merton, Mean-Variance, Buy-Hold
    5. **Analysis**: Crisis period stress testing, regime-conditional performance

    #### 📚 References

    - Merton, R.C. (1969) - "Lifetime Portfolio Selection"
    - Moody & Saffell (2001) - "Learning to Trade via Direct RL"
    - Modern portfolio theory and Deep RL research

    #### 🔗 Links

    - **GitHub**: [mohin-io/deep-rl-portfolio-allocation](https://github.com/mohin-io/deep-rl-portfolio-allocation)
    - **Author**: Mohin Hasin (@mohin-io)
    - **Email**: mohinhasin999@gmail.com

    #### ⚙️ Usage

    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Download data
    python src/data_pipeline/download.py

    # Train regime models
    python scripts/train_regime_models.py

    # Train DQN agent
    python scripts/train_dqn.py --episodes 1000

    # Launch dashboard
    streamlit run app/dashboard.py
    ```

    ---

    **Built with ❤️ for modern portfolio management**

    *Last Updated: 2025-10-03*
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Status")
st.sidebar.success("✅ Data Pipeline Complete")
st.sidebar.success("✅ Regime Detection Complete")
st.sidebar.warning("🔄 DQN Training In Progress")
st.sidebar.info("ℹ️ Dashboard v1.0")
