import os
import sys

# Ensure xrpl_bot is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from typing import List
import numpy as np
from xrpl_bot.backtest.strategies.base_strategy import (
    BaseStrategy, 
    RiskManagedStrategy, 
    EnsembleStrategy, 
    AdaptivePositionSizer, 
    TrailingStopStrategy
)

# Mocking a few basic strategies for the demo/test if needed, 
# but in real use we import from the directory.
class SimpleTrend(BaseStrategy):
    def next(self, bar_idx: int) -> Signal:
        from xrpl_bot.backtest.strategies.base_strategy import Signal
        close = self.data['close'][bar_idx]
        if bar_idx > 0 and close > self.data['close'][bar_idx-1]:
            return Signal('BUY', close, self.data['timestamp'][bar_idx], 0.5)
        return Signal('HOLD', close, self.data['timestamp'][bar_idx], 0.0)

class AdvancedHybridStrategy(BaseStrategy):
    """
    A sophisticated hybrid strategy that combines:
    1. Multiple technical indicators (Ensemble)
    2. Adaptive Position Sizing (based on ATR)
    3. Dynamic Risk Management (Stop Loss, Take Profit)
    """
    def __init__(self, 
                 sub_strategies: List[BaseStrategy],
                 voting_threshold: float = 0.6,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.5,
                 base_risk_pct: float = 0.01):
        super().__init__()
        
        from xrpl_bot.backtest.strategies.base_strategy import Signal
        self.Signal = Signal

        # 1. The core signal generator (Ensemble of sub-strategies)
        self.ensemble = EnsembleStrategy(sub_strategies, voting_threshold=voting_threshold)
        
        # 2. The risk manager (Wraps the ensemble to handle SL/TP)
        self.risk_manager = RiskManagedStrategy(
            self.ensemble, 
            stop_loss_pct=stop_loss_pct, 
            take_profit_pct=take_profit_pct
        )
        
        # 3. The position sizer
        self.sizer = AdaptivePositionSizer(
            base_risk_pct=base_risk_pct, 
            atr_period=atr_period
        )

    def init(self, data):
        super().init(data)
        self.ensemble.init(data)
        self.risk_manager.init(data)

    def next(self, bar_idx: int) -> Signal:
        return self.risk_manager.next(bar_idx)

    def get_position_size(self, current_price: float, atr: float, account_equity: float) -> float:
        return self.sizer.get_position_size(current_price, atr, account_equity)
