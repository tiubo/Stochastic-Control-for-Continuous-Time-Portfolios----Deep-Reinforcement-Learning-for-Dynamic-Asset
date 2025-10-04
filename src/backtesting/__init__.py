"""
Backtesting Framework

Comprehensive backtesting engine for portfolio strategies:
- BacktestEngine: Core simulation framework
- BacktestConfig: Configuration dataclass
- BacktestResults: Results container
- Strategy: Abstract base class for strategies
- Strategy Adapters: Connect baseline strategies to engine
"""

from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResults,
    Strategy,
    TradeRecord
)

from .strategy_adapters import (
    MertonStrategyAdapter,
    MeanVarianceAdapter,
    EqualWeightAdapter,
    BuyAndHoldAdapter,
    RiskParityAdapter,
    create_strategy_adapter
)

__all__ = [
    # Core engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResults',
    'Strategy',
    'TradeRecord',

    # Adapters
    'MertonStrategyAdapter',
    'MeanVarianceAdapter',
    'EqualWeightAdapter',
    'BuyAndHoldAdapter',
    'RiskParityAdapter',
    'create_strategy_adapter'
]
