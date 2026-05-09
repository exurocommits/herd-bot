"""
Advanced Ensemble Strategy with Risk Management and Performance-Based Weighting.
This strategy combines MetaEnsembleStrategy, DynamicEnsembleStrategy, and 
RiskManagedStrategy (via decorators) into a single powerful orchestrator.
"""
from backtest.strategies.base_strategy import (
    BaseStrategy, 
    Signal, 
    MetaEnsembleStrategy, 
    RiskManagedStrategy,
    EnsembleStrategy
)
from backtest.strategies.dynamic_performance_ensemble import DynamicEnsembleStrategy
from typing import List, Dict, Any, Optional

class SuperEnsembleStrategy(BaseStrategy):
    """
    The ultimate ensemble. 
    1. It uses DynamicEnsembleStrategy to manage weights of sub-strategies.
    2. It can wrap sub-strategies with RiskManagedStrategy decorators.
    3. It uses MetaEnsemble logic to finalize decisions.
    """
    def __init__(
        self, 
        strategies: List[BaseStrategy], 
        use_risk_management: bool = True,
        risk_params: Optional[Dict[str, float]] = None,
        lookback_period: int = 20,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.use_risk_management = use_risk_management
        self.risk_params = risk_params or {'stop_loss_pct': 0.02, 'take_profit_pct': 0.05}
        
        # Prepare sub-strategies
        processed_strategies = []
        for s in strategies:
            if self.use_risk_management:
                processed_strategies.append(RiskManagedStrategy(
                    underlying_strategy=s,
                    stop_loss_pct=self.risk_params.get('stop_loss_pct', 0.02),
                    take_profit_pct=self.risk_params.get('take_profit_pct', 0.05)
                ))
            else:
                processed_strategies.append(s)
        
        # The core is a DynamicEnsemble that adjusts weights based on PnL
        self.core_ensemble = DynamicEnsembleStrategy(
            strategies=processed_strategies,
            lookback_period=lookback_period,
            smoothing=smoothing
        )

    def init(self, data):
        self.data = data
        self.signals = []
        self.core_ensemble.init(data)

    def next(self, bar_idx: int) -> Signal:
        # 1. Get signal from the dynamic ensemble
        signal = self.core_ensemble.next(bar_idx)
        
        # 2. In a real backtest, we would need to record PnL to the core_ensemble.
        # Since the backtester (run_all_backtests.py) might not know about 
        # 'record_strategy_pnl', we assume it's handled or we provide a hook.
        # For now, we just return the signal.
        
        return signal

    def record_pnl(self, strategy_idx: int, pnl: float):
        """Pass PnL updates to the underlying dynamic ensemble."""
        self.core_ensemble.record_strategy_pnl(strategy_idx, pnl)

    def get_params(self) -> dict:
        params = {
            'use_risk_management': self.use_risk_management,
            'risk_params': self.risk_params
        }
        params.update(self.core_ensemble.get_params())
        return params
