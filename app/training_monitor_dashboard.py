"""
Real-Time Training Monitor Dashboard

Live monitoring dashboard for RL agent training featuring:
- Real-time metric updates
- Multi-agent comparison
- Performance tracking
- Hyperparameter display
- Training diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Training Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .status-running {
        color: #2ecc71;
    }
    .status-stopped {
        color: #e74c3c;
    }
    .status-pending {
        color: #f39c12;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=5)  # Cache for 5 seconds for real-time updates
def load_training_stats(filepath: str) -> pd.DataFrame:
    """Load training statistics with caching."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        return None


def load_agent_config(config_path: str) -> dict:
    """Load agent configuration."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except:
        return {}


def calculate_training_speed(stats: pd.DataFrame, window: int = 100) -> float:
    """Calculate training speed (episodes/hour)."""
    if len(stats) < window:
        return 0.0

    recent_stats = stats.tail(window)
    time_range = (recent_stats.index[-1] - recent_stats.index[0]) / 60  # in hours (approx)

    if time_range > 0:
        return window / time_range
    return 0.0


def main():
    """Main training monitor dashboard."""

    # Header
    st.title("🎓 Real-Time Training Monitor")

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Agent",
        options=['PPO', 'Prioritized DQN', 'Vanilla DQN'],
        index=0
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

    if auto_refresh:
        time.sleep(5)
        st.rerun()

    # Run directory
    run_dir = st.sidebar.text_input(
        "Run Directory",
        value=f"models/{model_type.lower().replace(' ', '_')}_optimized"
    )

    stats_path = Path(run_dir) / "training_stats.csv"

    # Load data
    stats = load_training_stats(str(stats_path))

    if stats is None or len(stats) == 0:
        st.warning(f"No training data found at: {stats_path}")
        st.info("Start training to see real-time metrics here!")
        return

    # ═══════════════════════════════════════════════════════════════
    # MAIN METRICS DISPLAY
    # ═══════════════════════════════════════════════════════════════

    # Status indicator
    last_update = datetime.now().strftime("%H:%M:%S")
    st.success(f"✅ Training Active | Last Update: {last_update}")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Episodes",
            f"{len(stats):,}",
            delta=f"+{min(100, len(stats))}" if len(stats) > 100 else None
        )

    with col2:
        mean_reward = stats['episode_reward'].tail(100).mean()
        prev_mean = stats['episode_reward'].iloc[-200:-100].mean() if len(stats) > 200 else mean_reward
        delta_reward = mean_reward - prev_mean

        st.metric(
            "Mean Reward (100ep)",
            f"{mean_reward:.2f}",
            delta=f"{delta_reward:+.2f}"
        )

    with col3:
        best_reward = stats['episode_reward'].max()
        st.metric(
            "Best Reward",
            f"{best_reward:.2f}"
        )

    with col4:
        if 'episode_length' in stats.columns:
            mean_length = stats['episode_length'].tail(100).mean()
            st.metric(
                "Mean Length (100ep)",
                f"{mean_length:.0f}"
            )

    with col5:
        speed = calculate_training_speed(stats)
        st.metric(
            "Speed",
            f"{speed:.1f} ep/hr"
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # TRAINING CURVES
    # ═══════════════════════════════════════════════════════════════

    st.subheader("📈 Training Progress")

    # Create subplot with rewards and lengths
    if 'episode_length' in stats.columns:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Episode Rewards', 'Episode Lengths'),
            vertical_spacing=0.1
        )
    else:
        fig = go.Figure()

    # Episode rewards
    # Raw values (transparent)
    fig.add_trace(
        go.Scatter(
            y=stats['episode_reward'],
            mode='lines',
            name='Reward',
            line=dict(color='lightblue', width=1),
            opacity=0.3,
            showlegend=True
        ),
        row=1, col=1 if 'episode_length' in stats.columns else None
    )

    # Moving averages
    for window in [10, 50, 100]:
        if len(stats) >= window:
            ma = stats['episode_reward'].rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    y=ma,
                    mode='lines',
                    name=f'MA-{window}',
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1, col=1 if 'episode_length' in stats.columns else None
            )

    # Episode lengths
    if 'episode_length' in stats.columns:
        fig.add_trace(
            go.Scatter(
                y=stats['episode_length'],
                mode='lines',
                name='Length',
                line=dict(color='lightgreen', width=1),
                opacity=0.3,
                showlegend=True
            ),
            row=2, col=1
        )

        ma_length = stats['episode_length'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                y=ma_length,
                mode='lines',
                name='MA-50',
                line=dict(color='green', width=2),
                showlegend=True
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Episode", row=2 if 'episode_length' in stats.columns else 1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)

    if 'episode_length' in stats.columns:
        fig.update_yaxes(title_text="Steps", row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=700 if 'episode_length' in stats.columns else 400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # DETAILED STATISTICS
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")
    st.subheader("📊 Detailed Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Reward Statistics**")

        reward_stats = {
            'Mean': stats['episode_reward'].mean(),
            'Std Dev': stats['episode_reward'].std(),
            'Min': stats['episode_reward'].min(),
            'Max': stats['episode_reward'].max(),
            'Median': stats['episode_reward'].median(),
            '25th Percentile': stats['episode_reward'].quantile(0.25),
            '75th Percentile': stats['episode_reward'].quantile(0.75)
        }

        for k, v in reward_stats.items():
            st.metric(k, f"{v:.2f}")

    with col2:
        if 'episode_length' in stats.columns:
            st.markdown("**Episode Length Statistics**")

            length_stats = {
                'Mean': stats['episode_length'].mean(),
                'Std Dev': stats['episode_length'].std(),
                'Min': stats['episode_length'].min(),
                'Max': stats['episode_length'].max(),
                'Median': stats['episode_length'].median()
            }

            for k, v in length_stats.items():
                st.metric(k, f"{v:.0f}")

    # ═══════════════════════════════════════════════════════════════
    # DISTRIBUTION ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")
    st.subheader("📊 Reward Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Histogram
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=stats['episode_reward'],
            nbinsx=50,
            name='Rewards',
            marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
        ))

        fig_hist.update_layout(
            title="Reward Distribution",
            xaxis_title="Reward",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("**Key Insights**")

        # Trend analysis
        recent_mean = stats['episode_reward'].tail(100).mean()
        overall_mean = stats['episode_reward'].mean()

        if recent_mean > overall_mean:
            trend = "📈 Improving"
            color = "green"
        else:
            trend = "📉 Declining"
            color = "red"

        st.markdown(f"**Trend:** :{color}[{trend}]")

        # Variance analysis
        recent_std = stats['episode_reward'].tail(100).std()
        overall_std = stats['episode_reward'].std()

        if recent_std < overall_std:
            stability = "✅ More Stable"
        else:
            stability = "⚠️ More Variable"

        st.markdown(f"**Stability:** {stability}")

        # Convergence estimate
        if len(stats) > 200:
            last_100 = stats['episode_reward'].tail(100).mean()
            prev_100 = stats['episode_reward'].iloc[-200:-100].mean()
            improvement = abs(last_100 - prev_100) / abs(prev_100) * 100

            if improvement < 5:
                convergence = "✅ Converging"
            else:
                convergence = "🔄 Still Learning"

            st.markdown(f"**Convergence:** {convergence}")
            st.markdown(f"**Improvement:** {improvement:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # RECENT EPISODES TABLE
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")
    st.subheader("📋 Recent Episodes")

    n_recent = st.slider("Number of recent episodes", 10, 100, 20)

    recent_stats = stats.tail(n_recent).copy()
    recent_stats.index = range(len(stats) - n_recent, len(stats))

    st.dataframe(
        recent_stats.style.format({
            'episode_reward': '{:.3f}',
            'episode_length': '{:.0f}' if 'episode_length' in recent_stats.columns else '{:.0f}'
        }).background_gradient(cmap='RdYlGn', subset=['episode_reward']),
        use_container_width=True
    )

    # ═══════════════════════════════════════════════════════════════
    # EXPORT DATA
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")
    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Download CSV
        csv = stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Stats (CSV)",
            data=csv,
            file_name=f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # Download summary
        summary = {
            'Total Episodes': len(stats),
            'Mean Reward': stats['episode_reward'].mean(),
            'Best Reward': stats['episode_reward'].max(),
            'Final 100 Mean': stats['episode_reward'].tail(100).mean(),
            'Training Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        summary_json = json.dumps(summary, indent=2)
        st.download_button(
            label="📥 Download Summary (JSON)",
            data=summary_json,
            file_name=f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Real-Time Training Monitor | Updates every 5 seconds (if auto-refresh enabled)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
