"""
Multi-Agent Volatility Management System
Specialized agents for detecting, forecasting, and countering market volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class VolatilityRegime(Enum):
    """Market volatility regimes"""
    EXTREME_LOW = 0      # VIX < 12, very calm
    LOW = 1              # VIX 12-16, normal calm
    NORMAL = 2           # VIX 16-20, average
    ELEVATED = 3         # VIX 20-30, heightened
    HIGH = 4             # VIX 30-40, stressed
    EXTREME_HIGH = 5     # VIX > 40, panic


class MarketRegime(Enum):
    """Broader market regimes"""
    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    BEAR_LOW_VOL = 2
    BEAR_HIGH_VOL = 3
    SIDEWAYS = 4
    CRISIS = 5


@dataclass
class VolatilitySignal:
    """Signal from volatility detection agent"""
    regime: VolatilityRegime
    current_vol: float
    forecasted_vol: float
    confidence: float
    alert_level: str  # 'green', 'yellow', 'red'
    timestamp: pd.Timestamp


@dataclass
class RiskSignal:
    """Signal from risk management agent"""
    max_drawdown_alert: bool
    var_breach: bool
    correlation_spike: bool
    leverage_warning: bool
    recommended_cash: float
    timestamp: pd.Timestamp


class VolatilityDetectionAgent:
    """
    Agent specialized in real-time volatility detection and classification.
    Uses multiple volatility measures and regime detection.
    """

    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.volatility_history = []
        self.regime_history = []

    def detect_regime(self, returns: pd.Series, vix: Optional[float] = None) -> VolatilityRegime:
        """
        Detect current volatility regime using multiple signals.

        Args:
            returns: Recent asset returns
            vix: VIX index value if available

        Returns:
            Current volatility regime
        """
        # Calculate realized volatility (annualized)
        realized_vol = returns.std() * np.sqrt(252) * 100

        # Use VIX if available, otherwise use realized vol
        if vix is not None:
            volatility_level = vix
        else:
            volatility_level = realized_vol

        # Classify regime
        if volatility_level < 12:
            return VolatilityRegime.EXTREME_LOW
        elif volatility_level < 16:
            return VolatilityRegime.LOW
        elif volatility_level < 20:
            return VolatilityRegime.NORMAL
        elif volatility_level < 30:
            return VolatilityRegime.ELEVATED
        elif volatility_level < 40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME_HIGH

    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics"""
        return {
            'realized_vol': returns.std() * np.sqrt(252) * 100,
            'downside_vol': returns[returns < 0].std() * np.sqrt(252) * 100 if len(returns[returns < 0]) > 0 else 0,
            'parkinson_vol': self._parkinson_volatility(returns),
            'ewma_vol': self._ewma_volatility(returns),
            'vol_of_vol': returns.rolling(5).std().std() * np.sqrt(252) * 100
        }

    def _parkinson_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """Parkinson high-low volatility estimator"""
        # Simplified version using returns range
        return returns.rolling(window).apply(lambda x: (x.max() - x.min())).std() * np.sqrt(252) * 100

    def _ewma_volatility(self, returns: pd.Series, span: int = 20) -> float:
        """Exponentially weighted moving average volatility"""
        return returns.ewm(span=span).std().iloc[-1] * np.sqrt(252) * 100

    def generate_signal(self, returns: pd.Series, vix: Optional[float] = None) -> VolatilitySignal:
        """Generate comprehensive volatility signal"""
        regime = self.detect_regime(returns, vix)
        metrics = self.calculate_volatility_metrics(returns)

        # Forecast next period volatility using EWMA
        forecasted_vol = metrics['ewma_vol'] * 1.05  # Simple forecast

        # Calculate confidence based on regime persistence
        confidence = 0.7 if len(self.regime_history) > 0 and self.regime_history[-1] == regime else 0.5

        # Determine alert level
        if regime in [VolatilityRegime.EXTREME_HIGH, VolatilityRegime.HIGH]:
            alert_level = 'red'
        elif regime == VolatilityRegime.ELEVATED:
            alert_level = 'yellow'
        else:
            alert_level = 'green'

        self.regime_history.append(regime)

        return VolatilitySignal(
            regime=regime,
            current_vol=metrics['realized_vol'],
            forecasted_vol=forecasted_vol,
            confidence=confidence,
            alert_level=alert_level,
            timestamp=pd.Timestamp.now()
        )


class RiskManagementAgent:
    """
    Agent specialized in dynamic risk management and position sizing.
    Adapts portfolio constraints based on market conditions.
    """

    def __init__(
        self,
        max_drawdown_threshold: float = 0.20,
        var_confidence: float = 0.95,
        max_leverage: float = 1.0
    ):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.var_confidence = var_confidence
        self.max_leverage = max_leverage
        self.peak_value = 0.0

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def assess_risk(
        self,
        portfolio_value: float,
        returns: pd.Series,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> RiskSignal:
        """
        Comprehensive risk assessment.

        Args:
            portfolio_value: Current portfolio value
            returns: Recent portfolio returns
            correlation_matrix: Asset correlation matrix

        Returns:
            Risk signal with recommendations
        """
        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calculate current drawdown
        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
        max_drawdown_alert = abs(current_drawdown) > self.max_drawdown_threshold

        # VaR analysis
        var_95 = self.calculate_var(returns, 0.95)
        current_return = returns.iloc[-1] if len(returns) > 0 else 0
        var_breach = current_return < var_95

        # Correlation spike detection
        correlation_spike = False
        if correlation_matrix is not None:
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            correlation_spike = avg_correlation > 0.7  # High correlation = systemic risk

        # Recommended cash allocation based on risk
        if max_drawdown_alert or var_breach or correlation_spike:
            recommended_cash = 0.30  # 30% cash in high risk
        else:
            recommended_cash = 0.05  # 5% cash normally

        return RiskSignal(
            max_drawdown_alert=max_drawdown_alert,
            var_breach=var_breach,
            correlation_spike=correlation_spike,
            leverage_warning=False,  # Can be extended
            recommended_cash=recommended_cash,
            timestamp=pd.Timestamp.now()
        )


class RegimeDetectionAgent:
    """
    Agent specialized in detecting broader market regimes.
    Combines trend, volatility, and momentum signals.
    """

    def __init__(self, lookback_short: int = 20, lookback_long: int = 60):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def detect_trend(self, prices: pd.Series) -> str:
        """Detect market trend using moving averages"""
        ma_short = prices.rolling(self.lookback_short).mean().iloc[-1]
        ma_long = prices.rolling(self.lookback_long).mean().iloc[-1]

        if ma_short > ma_long * 1.02:
            return 'bull'
        elif ma_short < ma_long * 0.98:
            return 'bear'
        else:
            return 'sideways'

    def detect_market_regime(
        self,
        prices: pd.Series,
        returns: pd.Series,
        vix: Optional[float] = None
    ) -> MarketRegime:
        """
        Detect comprehensive market regime.

        Args:
            prices: Asset prices
            returns: Asset returns
            vix: VIX index if available

        Returns:
            Market regime classification
        """
        trend = self.detect_trend(prices)
        vol = returns.std() * np.sqrt(252) * 100

        # Use VIX if available
        if vix is not None:
            vol = vix

        high_vol = vol > 25

        # Crisis detection (extreme volatility + negative returns)
        recent_return = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        if vol > 40 and recent_return < -0.10:
            return MarketRegime.CRISIS

        # Classify regime
        if trend == 'bull':
            return MarketRegime.BULL_HIGH_VOL if high_vol else MarketRegime.BULL_LOW_VOL
        elif trend == 'bear':
            return MarketRegime.BEAR_HIGH_VOL if high_vol else MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS


class AdaptiveRebalancingAgent:
    """
    Agent specialized in dynamic portfolio rebalancing.
    Adjusts rebalancing frequency and thresholds based on volatility.
    """

    def __init__(
        self,
        base_threshold: float = 0.05,
        min_days_between: int = 5,
        max_days_between: int = 30
    ):
        self.base_threshold = base_threshold
        self.min_days_between = min_days_between
        self.max_days_between = max_days_between
        self.last_rebalance_day = 0

    def should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        volatility_regime: VolatilityRegime,
        days_since_last: int
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.
        More aggressive rebalancing during high volatility.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            volatility_regime: Current volatility regime
            days_since_last: Days since last rebalance

        Returns:
            Whether to rebalance
        """
        # Calculate drift from target
        max_drift = np.max(np.abs(current_weights - target_weights))

        # Adjust threshold based on volatility
        if volatility_regime in [VolatilityRegime.EXTREME_HIGH, VolatilityRegime.HIGH]:
            threshold = self.base_threshold * 0.5  # More sensitive
            min_days = self.min_days_between // 2
        elif volatility_regime == VolatilityRegime.ELEVATED:
            threshold = self.base_threshold * 0.75
            min_days = self.min_days_between
        else:
            threshold = self.base_threshold
            min_days = self.min_days_between

        # Check conditions
        drift_exceeded = max_drift > threshold
        time_exceeded = days_since_last >= min_days

        return drift_exceeded and time_exceeded

    def calculate_optimal_weights(
        self,
        returns: pd.DataFrame,
        volatility_regime: VolatilityRegime,
        risk_signal: RiskSignal,
        method: str = 'risk_parity'
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights adapted to current regime.

        Args:
            returns: Asset returns DataFrame
            volatility_regime: Current volatility regime
            risk_signal: Risk management signal
            method: Allocation method

        Returns:
            Optimal weights array
        """
        n_assets = returns.shape[1]

        if method == 'risk_parity':
            # Risk parity with regime adjustment
            volatilities = returns.std() * np.sqrt(252)
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()

        elif method == 'minimum_variance':
            # Minimum variance portfolio
            cov_matrix = returns.cov() * 252
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(n_assets)
            weights = inv_cov @ ones / (ones @ inv_cov @ ones)

        else:  # equal_weight
            weights = np.ones(n_assets) / n_assets

        # Adjust for volatility regime
        if volatility_regime in [VolatilityRegime.EXTREME_HIGH, VolatilityRegime.HIGH]:
            # Reduce risk exposure, increase cash
            cash_weight = risk_signal.recommended_cash
            weights = weights * (1 - cash_weight)

        # Ensure non-negative and sum to 1
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return weights


class VolatilityForecastingAgent:
    """
    Agent specialized in volatility forecasting using GARCH-like models.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def forecast_garch(self, returns: pd.Series, horizon: int = 5) -> np.ndarray:
        """
        Simple GARCH(1,1) inspired volatility forecast.

        Args:
            returns: Historical returns
            horizon: Forecast horizon in days

        Returns:
            Forecasted volatility for each day
        """
        # Parameters (simplified, not estimated)
        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        # Current variance
        current_var = returns.iloc[-20:].var()

        # Forecast
        forecasts = []
        var_forecast = current_var

        for _ in range(horizon):
            var_forecast = omega + alpha * (returns.iloc[-1] ** 2) + beta * var_forecast
            forecasts.append(np.sqrt(var_forecast) * np.sqrt(252) * 100)

        return np.array(forecasts)

    def forecast_ewma(self, returns: pd.Series, horizon: int = 5, lambda_param: float = 0.94) -> np.ndarray:
        """
        Exponentially weighted moving average volatility forecast.

        Args:
            returns: Historical returns
            horizon: Forecast horizon
            lambda_param: Decay parameter

        Returns:
            Forecasted volatility
        """
        # Calculate EWMA variance
        squared_returns = returns ** 2
        ewma_var = squared_returns.ewm(alpha=1-lambda_param).mean().iloc[-1]

        # Assume persistence
        forecasts = [np.sqrt(ewma_var) * np.sqrt(252) * 100] * horizon

        return np.array(forecasts)


class AgentCoordinator:
    """
    Master coordinator that integrates signals from all agents.
    Makes final portfolio decisions based on consensus.
    """

    def __init__(self):
        self.volatility_agent = VolatilityDetectionAgent()
        self.risk_agent = RiskManagementAgent()
        self.regime_agent = RegimeDetectionAgent()
        self.rebalancing_agent = AdaptiveRebalancingAgent()
        self.forecasting_agent = VolatilityForecastingAgent()

    def get_portfolio_recommendation(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        current_weights: np.ndarray,
        portfolio_value: float,
        vix: Optional[float] = None,
        days_since_rebalance: int = 0
    ) -> Dict[str, any]:
        """
        Coordinate all agents to produce comprehensive portfolio recommendation.

        Args:
            prices: Asset prices DataFrame
            returns: Asset returns DataFrame
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value
            vix: VIX index value
            days_since_rebalance: Days since last rebalance

        Returns:
            Comprehensive recommendation dictionary
        """
        # Get signals from all agents
        portfolio_returns = (returns * current_weights).sum(axis=1)

        vol_signal = self.volatility_agent.generate_signal(portfolio_returns, vix)
        risk_signal = self.risk_agent.assess_risk(
            portfolio_value,
            portfolio_returns,
            returns.corr()
        )
        market_regime = self.regime_agent.detect_market_regime(
            prices.iloc[:, 0],  # Use first asset as proxy
            portfolio_returns,
            vix
        )

        # Volatility forecast
        vol_forecast = self.forecasting_agent.forecast_garch(portfolio_returns, horizon=5)

        # Rebalancing decision
        target_weights = self.rebalancing_agent.calculate_optimal_weights(
            returns,
            vol_signal.regime,
            risk_signal,
            method='risk_parity'
        )

        should_rebalance = self.rebalancing_agent.should_rebalance(
            current_weights,
            target_weights,
            vol_signal.regime,
            days_since_rebalance
        )

        return {
            'volatility_signal': vol_signal,
            'risk_signal': risk_signal,
            'market_regime': market_regime,
            'volatility_forecast': vol_forecast,
            'should_rebalance': should_rebalance,
            'recommended_weights': target_weights,
            'alert_level': vol_signal.alert_level,
            'recommended_cash': risk_signal.recommended_cash,
            'timestamp': pd.Timestamp.now()
        }
