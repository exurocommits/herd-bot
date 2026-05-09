from typing import List
import numpy as np
from xrpl_bot.backtest.strategies.base_strategy import (
    BaseStrategy, 
    RiskManagedStrategy, 
    EnsembleStrategy, 
    AdaptivePositionSizer, 
    TrailingStopStrategy
)

class AdvancedHybridStrategy(BaseStrategy):
    """
    A sophisticated hybrid strategy that combines:
    1. Multiple technical indicators (Ensemble)
    2. Adaptive Position Sizing (based on ATR)
    3. Dynamic Risk Management (Trailing Stops & SL/TP)
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
        
        # 1. The core signal generator (Ensemble of sub-strategies)
        self.ensemble = EnsembleStrategy(sub_strategies, voting_threshold=voting_threshold)
        
        # 2. The risk manager (Wraps the ensemble to handle SL/TP)
        self.risk_manager = RiskManagedStrategy(
            self.ensemble, 
            stop_loss_pct=stop_loss_pct, 
            take_profit_pct=take_profit_pct
        )
        
        # 3. The trailing stop component (can be integrated into the risk manager or used separately)
        self.trailing_stop = TrailingStopStrategy(
            atr_multiplier=atr_multiplier, 
            atr_period=atr_period
        )
        
        # 4. The position sizer
        self.sizer = AdaptivePositionSizer(
            base_risk_pct=base_risk_pct, 
            atr_period=atr_period
        )

    def init(self, data):
        super().init(data)
        self.ensemble.init(data)
        self.risk_manager.init(data)
        self.trailing_stop.init(data)

    def next(self, bar_idx: int) -> Signal:
        # In a real backtest, we would use the signal from risk_manager 
        # but we can also inject trailing stop logic here if we modify RiskManagedStrategy.
        # For now, we return the risk-managed signal.
        return self.risk_manager.next(bar_idx)

    def get_position_size(self, current_price: float, atr: float, account_equity: float) -> float:
        return self.sizer.get_position_size(current_price, atr, account_equity)
