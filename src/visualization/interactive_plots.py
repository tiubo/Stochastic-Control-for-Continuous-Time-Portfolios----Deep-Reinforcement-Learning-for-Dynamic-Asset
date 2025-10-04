"""
Interactive Visualization Module using Plotly

Creates interactive, publication-quality visualizations for:
- Portfolio performance analysis
- Regime detection visualization
- Risk-return analysis
- Drawdown analysis
- Training progress monitoring
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class InteractiveVisualizer:
    """Create interactive visualizations using Plotly."""

    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize visualizer.

        Args:
            theme: Plotly theme ('plotly', 'plotly_dark', 'plotly_white', 'ggplot2', 'seaborn')
        """
        self.theme = theme
        self.colors = {
            'bull': '#2ecc71',      # Green
            'bear': '#e74c3c',      # Red
            'volatile': '#f39c12',  # Orange
            'dqn': '#3498db',       # Blue
            'ppo': '#9b59b6',       # Purple
            'merton': '#95a5a6',    # Gray
            'spy': '#1abc9c'        # Teal
        }

    def plot_portfolio_evolution(
        self,
        strategies: Dict[str, pd.Series],
        title: str = "Portfolio Value Evolution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot portfolio value evolution for multiple strategies.

        Args:
            strategies: Dict of strategy_name -> portfolio_values (pd.Series with DatetimeIndex)
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Add trace for each strategy
        for name, values in strategies.items():
            color = self.colors.get(name.lower(), None)

            fig.add_trace(go.Scatter(
                x=values.index,
                y=values.values,
                mode='lines',
                name=name,
                line=dict(width=2, color=color),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Value: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_returns_distribution(
        self,
        strategies: Dict[str, pd.Series],
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot returns distribution comparison.

        Args:
            strategies: Dict of strategy_name -> returns (pd.Series)
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, returns in strategies.items():
            color = self.colors.get(name.lower(), None)

            fig.add_trace(go.Histogram(
                x=returns * 100,  # Convert to percentage
                name=name,
                opacity=0.7,
                marker=dict(color=color),
                nbinsx=50,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Return: %{x:.2f}%<br>' +
                             'Count: %{y}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Daily Returns (%)",
            yaxis_title="Frequency",
            template=self.theme,
            barmode='overlay',
            height=500
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_regime_colored_prices(
        self,
        prices: pd.Series,
        regimes: pd.Series,
        regime_names: Dict[int, str] = {0: 'Bull', 1: 'Bear', 2: 'Volatile'},
        title: str = "Asset Prices by Market Regime",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot asset prices colored by market regime.

        Args:
            prices: Price series with DatetimeIndex
            regimes: Regime labels (0, 1, 2)
            regime_names: Mapping of regime ID to name
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Align prices and regimes
        df = pd.DataFrame({'price': prices, 'regime': regimes})
        df = df.dropna()

        # Plot each regime separately for color coding
        for regime_id, regime_name in regime_names.items():
            mask = df['regime'] == regime_id
            regime_data = df[mask]

            if len(regime_data) > 0:
                color = self.colors.get(regime_name.lower(), '#7f8c8d')

                fig.add_trace(go.Scatter(
                    x=regime_data.index,
                    y=regime_data['price'],
                    mode='lines',
                    name=regime_name,
                    line=dict(color=color, width=2),
                    hovertemplate='<b>' + regime_name + '</b><br>' +
                                 'Date: %{x|%Y-%m-%d}<br>' +
                                 'Price: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template=self.theme,
            hovermode='x unified',
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_drawdown_comparison(
        self,
        strategies: Dict[str, pd.Series],
        title: str = "Drawdown Comparison",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot drawdown comparison for multiple strategies.

        Args:
            strategies: Dict of strategy_name -> portfolio_values
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, values in strategies.items():
            # Calculate drawdown
            running_max = values.cummax()
            drawdown = (values - running_max) / running_max

            color = self.colors.get(name.lower(), None)

            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,  # Convert to percentage
                mode='lines',
                name=name,
                fill='tozeroy',
                line=dict(color=color, width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Drawdown: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.theme,
            hovermode='x unified',
            height=500
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_risk_return_scatter(
        self,
        metrics: pd.DataFrame,
        title: str = "Risk-Return Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot risk-return scatter with Sharpe ratio annotations.

        Args:
            metrics: DataFrame with columns ['annualized_return', 'volatility', 'sharpe_ratio']
                     and strategy names as index
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for strategy in metrics.index:
            ret = metrics.loc[strategy, 'annualized_return'] * 100
            vol = metrics.loc[strategy, 'volatility'] * 100
            sharpe = metrics.loc[strategy, 'sharpe_ratio']

            color = self.colors.get(strategy.lower(), '#7f8c8d')

            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers+text',
                name=strategy,
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                text=[strategy],
                textposition='top center',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Return: %{y:.2f}%<br>' +
                             'Volatility: %{x:.2f}%<br>' +
                             f'Sharpe: {sharpe:.3f}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Volatility (Annualized %)",
            yaxis_title="Return (Annualized %)",
            template=self.theme,
            showlegend=False,
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_rolling_metrics(
        self,
        strategies: Dict[str, pd.Series],
        window: int = 252,
        metric: str = 'sharpe',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot rolling performance metrics.

        Args:
            strategies: Dict of strategy_name -> returns
            window: Rolling window size (days)
            metric: Metric to plot ('sharpe', 'volatility', 'return')
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, returns in strategies.items():
            if metric == 'sharpe':
                # Rolling Sharpe ratio
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                values = (rolling_mean / rolling_std) * np.sqrt(252)
                ylabel = "Sharpe Ratio"
            elif metric == 'volatility':
                # Rolling volatility
                values = returns.rolling(window).std() * np.sqrt(252) * 100
                ylabel = "Volatility (%)"
            else:  # return
                # Rolling return
                values = returns.rolling(window).mean() * 252 * 100
                ylabel = "Annualized Return (%)"

            color = self.colors.get(name.lower(), None)

            fig.add_trace(go.Scatter(
                x=values.index,
                y=values.values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             f'{ylabel}: ' + '%{y:.2f}<br>' +
                             '<extra></extra>'
            ))

        if title is None:
            title = f"Rolling {metric.capitalize()} ({window}-day window)"

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title=ylabel,
            template=self.theme,
            hovermode='x unified',
            height=500
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_training_progress(
        self,
        training_stats: pd.DataFrame,
        metrics: List[str] = ['episode_reward', 'episode_length'],
        title: str = "Training Progress",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot training progress with multiple metrics.

        Args:
            training_stats: DataFrame with training metrics
            metrics: List of metric column names to plot
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        # Create subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )

        for i, metric in enumerate(metrics, 1):
            if metric in training_stats.columns:
                # Raw values
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(training_stats))),
                        y=training_stats[metric],
                        mode='lines',
                        name=f'{metric} (raw)',
                        line=dict(color='lightblue', width=1),
                        opacity=0.3,
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

                # Moving average
                window = min(100, len(training_stats) // 10)
                ma = training_stats[metric].rolling(window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(training_stats))),
                        y=ma,
                        mode='lines',
                        name=f'{metric} (MA-{window})',
                        line=dict(color='blue', width=2),
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

        fig.update_xaxes(title_text="Episode", row=n_metrics, col=1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=self.theme,
            height=300 * n_metrics,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_correlation_matrix(
        self,
        returns: pd.DataFrame,
        title: str = "Asset Return Correlations",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            returns: DataFrame of returns for multiple assets
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        corr = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=self.theme,
            height=600,
            width=700
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_regime_transitions(
        self,
        transition_matrix: np.ndarray,
        regime_names: List[str] = ['Bull', 'Bear', 'Volatile'],
        title: str = "Regime Transition Probabilities",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot regime transition probability matrix.

        Args:
            transition_matrix: NxN transition probability matrix
            regime_names: Names for each regime
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=regime_names,
            y=regime_names,
            colorscale='Viridis',
            text=np.round(transition_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='From %{y} to %{x}<br>Probability: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            template=self.theme,
            height=500,
            width=600
        )

        if save_path:
            fig.write_html(save_path)

        return fig


def create_dashboard_plots(
    data: pd.DataFrame,
    output_dir: str = "docs/figures/interactive"
) -> Dict[str, go.Figure]:
    """
    Create all interactive dashboard plots.

    Args:
        data: Processed dataset with regimes
        output_dir: Directory to save HTML files

    Returns:
        Dictionary of figure_name -> plotly figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = InteractiveVisualizer(theme='plotly_dark')
    figures = {}

    # 1. Regime-colored price chart
    if 'price_SPY' in data.columns and 'regime_gmm' in data.columns:
        fig = viz.plot_regime_colored_prices(
            prices=data['price_SPY'],
            regimes=data['regime_gmm'],
            title="SPY Price by Market Regime (GMM)",
            save_path=str(output_path / "regime_colored_prices.html")
        )
        figures['regime_prices'] = fig

    # 2. Correlation matrix
    return_cols = [col for col in data.columns if col.startswith('return_')]
    if len(return_cols) > 0:
        returns = data[return_cols].dropna()
        fig = viz.plot_correlation_matrix(
            returns=returns,
            title="Asset Return Correlations",
            save_path=str(output_path / "correlation_matrix.html")
        )
        figures['correlation'] = fig

    print(f"Created {len(figures)} interactive plots in {output_dir}")
    return figures


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '.')

    # Load data
    data = pd.read_csv("data/processed/dataset_with_regimes.csv",
                      index_col=0, parse_dates=True)

    # Create plots
    figures = create_dashboard_plots(data)
    print(f"Generated {len(figures)} interactive visualizations")
