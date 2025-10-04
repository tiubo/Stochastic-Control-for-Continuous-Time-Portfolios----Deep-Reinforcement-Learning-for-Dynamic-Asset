"""
Advanced Analytics Dashboard

Comprehensive interactive dashboard for portfolio analysis featuring:
- Real-time performance monitoring
- Multi-strategy comparison
- Risk analytics
- Regime analysis
- Training progress tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.interactive_plots import InteractiveVisualizer
from src.backtesting.performance_benchmark import PerformanceMetrics, StrategyComparison

# Page configuration
st.set_page_config(
    page_title="RL Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #2ecc71;
    }
    .negative {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset with caching."""
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def load_training_stats(stats_path: str) -> pd.DataFrame:
    """Load training statistics."""
    try:
        stats = pd.read_csv(stats_path)
        return stats
    except Exception as e:
        st.warning(f"Training stats not found: {e}")
        return None


def calculate_portfolio_metrics(returns: pd.Series) -> dict:
    """Calculate comprehensive portfolio metrics."""
    if len(returns) == 0:
        return {}

    metrics = {
        'Total Return': f"{(returns + 1).prod() - 1:.2%}",
        'Annualized Return': f"{returns.mean() * 252:.2%}",
        'Volatility': f"{returns.std() * np.sqrt(252):.2%}",
        'Sharpe Ratio': f"{PerformanceMetrics.sharpe_ratio(returns):.3f}",
        'Sortino Ratio': f"{PerformanceMetrics.sortino_ratio(returns):.3f}",
        'Max Drawdown': f"{PerformanceMetrics.max_drawdown(returns):.2%}",
        'Win Rate': f"{PerformanceMetrics.win_rate(returns):.2%}",
        'Profit Factor': f"{PerformanceMetrics.profit_factor(returns):.2f}"
    }

    return metrics


def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-header">📊 Deep RL Portfolio Analytics Dashboard</h1>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Data loading
    data_path = st.sidebar.text_input(
        "Dataset Path",
        value="data/processed/dataset_with_regimes.csv"
    )

    data = load_data(data_path)

    if data is None:
        st.error("Failed to load data. Please check the file path.")
        return

    # Theme selection
    theme = st.sidebar.selectbox(
        "Chart Theme",
        options=['plotly_dark', 'plotly', 'plotly_white', 'seaborn'],
        index=0
    )

    viz = InteractiveVisualizer(theme=theme)

    # Date range selector
    st.sidebar.subheader("📅 Date Range")
    min_date = data.index.min().date()
    max_date = data.index.max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter data by date range
    if len(date_range) == 2:
        mask = (data.index.date >= date_range[0]) & (data.index.date <= date_range[1])
        filtered_data = data[mask]
    else:
        filtered_data = data

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Portfolio Overview",
        "🎯 Regime Analysis",
        "⚖️ Risk Analytics",
        "📊 Strategy Comparison",
        "🎓 Training Monitor"
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: Portfolio Overview
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.header("Portfolio Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics for SPY (as example)
        if 'return_SPY' in filtered_data.columns:
            spy_returns = filtered_data['return_SPY'].dropna()

            with col1:
                total_return = (spy_returns + 1).prod() - 1
                st.metric(
                    "Total Return (SPY)",
                    f"{total_return:.2%}",
                    delta=f"{spy_returns.iloc[-1]:.2%}" if len(spy_returns) > 0 else "0%"
                )

            with col2:
                sharpe = PerformanceMetrics.sharpe_ratio(spy_returns)
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")

            with col3:
                vol = spy_returns.std() * np.sqrt(252)
                st.metric("Volatility", f"{vol:.2%}")

            with col4:
                max_dd = PerformanceMetrics.max_drawdown(spy_returns)
                st.metric("Max Drawdown", f"{max_dd:.2%}")

        st.markdown("---")

        # Asset price evolution
        st.subheader("📈 Asset Price Evolution")

        price_cols = [col for col in filtered_data.columns if col.startswith('price_')]
        if len(price_cols) > 0:
            selected_assets = st.multiselect(
                "Select Assets",
                options=price_cols,
                default=price_cols[:min(4, len(price_cols))]
            )

            if selected_assets:
                # Create normalized price chart
                fig = go.Figure()

                for col in selected_assets:
                    # Normalize to 100
                    normalized = filtered_data[col] / filtered_data[col].iloc[0] * 100

                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized,
                        mode='lines',
                        name=col.replace('price_', ''),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Date: %{x|%Y-%m-%d}<br>' +
                                     'Value: %{y:.2f}<br>' +
                                     '<extra></extra>'
                    ))

                fig.update_layout(
                    title="Normalized Asset Prices (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Value",
                    template=theme,
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

        # Returns distribution
        st.subheader("📊 Returns Distribution")

        return_cols = [col for col in filtered_data.columns if col.startswith('return_')]
        if len(return_cols) > 0:
            selected_returns = st.selectbox(
                "Select Asset",
                options=return_cols,
                index=0
            )

            returns = filtered_data[selected_returns].dropna() * 100  # Convert to %

            # Create histogram with statistics
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))

            # Add normal distribution overlay
            mean = returns.mean()
            std = returns.std()
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
            normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50

            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig.update_layout(
                title=f"Returns Distribution - {selected_returns.replace('return_', '')}",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template=theme,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean", f"{mean:.3f}%")
            with col2:
                st.metric("Std Dev", f"{std:.3f}%")
            with col3:
                st.metric("Skewness", f"{returns.skew():.3f}")
            with col4:
                st.metric("Kurtosis", f"{returns.kurt():.3f}")
            with col5:
                st.metric("Min/Max", f"{returns.min():.2f}% / {returns.max():.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: Regime Analysis
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.header("Market Regime Analysis")

        # Regime selector
        regime_method = st.radio(
            "Regime Detection Method",
            options=['GMM', 'HMM'],
            horizontal=True
        )

        regime_col = f'regime_{"gmm" if regime_method == "GMM" else "hmm"}'

        if regime_col in filtered_data.columns:
            # Regime distribution
            st.subheader("📊 Regime Distribution")

            regime_counts = filtered_data[regime_col].value_counts().sort_index()
            regime_names = {0: 'Bull', 1: 'Bear', 2: 'Volatile'} if regime_method == 'GMM' else {0: 'Volatile', 1: 'Bull', 2: 'Bear'}

            col1, col2 = st.columns([1, 2])

            with col1:
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[regime_names[i] for i in regime_counts.index],
                    values=regime_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#2ecc71', '#e74c3c', '#f39c12']),
                    hovertemplate='<b>%{label}</b><br>' +
                                 'Count: %{value}<br>' +
                                 'Percentage: %{percent}<br>' +
                                 '<extra></extra>'
                )])

                fig_pie.update_layout(
                    title=f"{regime_method} Regime Distribution",
                    template=theme,
                    height=400
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Bar chart with percentages
                regime_pct = regime_counts / regime_counts.sum() * 100

                fig_bar = go.Figure(data=[go.Bar(
                    x=[regime_names[i] for i in regime_counts.index],
                    y=regime_pct.values,
                    marker=dict(color=['#2ecc71', '#e74c3c', '#f39c12']),
                    text=[f"{v:.1f}%" for v in regime_pct.values],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' +
                                 'Percentage: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                )])

                fig_bar.update_layout(
                    title="Regime Frequencies",
                    yaxis_title="Percentage (%)",
                    template=theme,
                    height=400
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            # Regime-colored price chart
            st.subheader("📈 Price Evolution by Regime")

            if 'price_SPY' in filtered_data.columns:
                fig_regime = viz.plot_regime_colored_prices(
                    prices=filtered_data['price_SPY'],
                    regimes=filtered_data[regime_col],
                    regime_names=regime_names,
                    title=f"SPY Price by Market Regime ({regime_method})"
                )

                st.plotly_chart(fig_regime, use_container_width=True)

            # Regime statistics
            st.subheader("📊 Performance by Regime")

            if 'return_SPY' in filtered_data.columns:
                regime_stats = []

                for regime_id, regime_name in regime_names.items():
                    mask = filtered_data[regime_col] == regime_id
                    regime_returns = filtered_data.loc[mask, 'return_SPY'].dropna()

                    if len(regime_returns) > 0:
                        regime_stats.append({
                            'Regime': regime_name,
                            'Observations': len(regime_returns),
                            'Mean Return (%)': regime_returns.mean() * 100,
                            'Volatility (%)': regime_returns.std() * 100,
                            'Sharpe Ratio': PerformanceMetrics.sharpe_ratio(regime_returns),
                            'Max Drawdown (%)': PerformanceMetrics.max_drawdown(regime_returns) * 100
                        })

                stats_df = pd.DataFrame(regime_stats)
                st.dataframe(stats_df.style.format({
                    'Mean Return (%)': '{:.3f}',
                    'Volatility (%)': '{:.3f}',
                    'Sharpe Ratio': '{:.3f}',
                    'Max Drawdown (%)': '{:.2f}'
                }), use_container_width=True)

        else:
            st.warning(f"Regime column '{regime_col}' not found in dataset")

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: Risk Analytics
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.header("Risk Analytics")

        if 'return_SPY' in filtered_data.columns:
            spy_returns = filtered_data['return_SPY'].dropna()

            # Risk metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                var_95 = PerformanceMetrics.value_at_risk(spy_returns, 0.95) * 100
                st.metric("VaR (95%)", f"{var_95:.3f}%")

            with col2:
                cvar_95 = PerformanceMetrics.conditional_var(spy_returns, 0.95) * 100
                st.metric("CVaR (95%)", f"{cvar_95:.3f}%")

            with col3:
                omega = PerformanceMetrics.omega_ratio(spy_returns)
                st.metric("Omega Ratio", f"{omega:.3f}")

            st.markdown("---")

            # Drawdown chart
            st.subheader("📉 Drawdown Analysis")

            if 'price_SPY' in filtered_data.columns:
                prices = filtered_data['price_SPY'].dropna()
                running_max = prices.cummax()
                drawdown = (prices - running_max) / running_max * 100

                fig_dd = go.Figure()

                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>' +
                                 'Drawdown: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ))

                fig_dd.update_layout(
                    title="SPY Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    template=theme,
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig_dd, use_container_width=True)

            # Rolling volatility
            st.subheader("📊 Rolling Volatility")

            window = st.slider("Rolling Window (days)", 30, 252, 60)

            rolling_vol = spy_returns.rolling(window).std() * np.sqrt(252) * 100

            fig_vol = go.Figure()

            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name=f'{window}-day Volatility',
                line=dict(color='orange', width=2),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>' +
                             'Volatility: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))

            fig_vol.update_layout(
                title=f"Rolling {window}-day Volatility",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                template=theme,
                height=400
            )

            st.plotly_chart(fig_vol, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: Strategy Comparison
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.header("Strategy Comparison")

        st.info("This section will display comparisons between DQN, PPO, and Merton strategies once models are trained.")

        # Placeholder for future implementation
        st.markdown("""
        **Coming Soon:**
        - Portfolio value evolution comparison
        - Risk-return scatter plot
        - Drawdown comparison
        - Statistical significance tests
        - Crisis period performance
        """)

    # ═══════════════════════════════════════════════════════════════
    # TAB 5: Training Monitor
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.header("Training Progress Monitor")

        training_path = st.text_input(
            "Training Stats Path",
            value="models/ppo_optimized/final_training_stats.csv"
        )

        stats = load_training_stats(training_path)

        if stats is not None and len(stats) > 0:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Episodes", len(stats))

            with col2:
                mean_reward = stats['episode_reward'].mean()
                st.metric("Mean Reward", f"{mean_reward:.2f}")

            with col3:
                final_reward = stats['episode_reward'].iloc[-100:].mean()
                st.metric("Final 100 Ep Reward", f"{final_reward:.2f}")

            with col4:
                improvement = ((final_reward - stats['episode_reward'].iloc[:100].mean()) /
                             abs(stats['episode_reward'].iloc[:100].mean()) * 100)
                st.metric("Improvement", f"{improvement:.1f}%")

            # Training curves
            st.subheader("📈 Training Curves")

            # Episode reward
            fig_reward = go.Figure()

            # Raw rewards (transparent)
            fig_reward.add_trace(go.Scatter(
                y=stats['episode_reward'],
                mode='lines',
                name='Episode Reward',
                line=dict(color='lightblue', width=1),
                opacity=0.3
            ))

            # Moving average
            ma_window = min(100, len(stats) // 10)
            ma_reward = stats['episode_reward'].rolling(ma_window).mean()

            fig_reward.add_trace(go.Scatter(
                y=ma_reward,
                mode='lines',
                name=f'MA-{ma_window}',
                line=dict(color='blue', width=2)
            ))

            fig_reward.update_layout(
                title="Episode Rewards",
                xaxis_title="Episode",
                yaxis_title="Reward",
                template=theme,
                height=400
            )

            st.plotly_chart(fig_reward, use_container_width=True)

            # Episode length (if available)
            if 'episode_length' in stats.columns:
                fig_length = go.Figure()

                fig_length.add_trace(go.Scatter(
                    y=stats['episode_length'],
                    mode='lines',
                    name='Episode Length',
                    line=dict(color='green', width=1),
                    opacity=0.5
                ))

                ma_length = stats['episode_length'].rolling(ma_window).mean()

                fig_length.add_trace(go.Scatter(
                    y=ma_length,
                    mode='lines',
                    name=f'MA-{ma_window}',
                    line=dict(color='darkgreen', width=2)
                ))

                fig_length.update_layout(
                    title="Episode Lengths",
                    xaxis_title="Episode",
                    yaxis_title="Steps",
                    template=theme,
                    height=300
                )

                st.plotly_chart(fig_length, use_container_width=True)

        else:
            st.info("No training statistics found. Train a model to see progress here.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Deep RL Portfolio Allocation Dashboard | Built with Streamlit & Plotly</p>
        <p>Data updates in real-time as models train</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
