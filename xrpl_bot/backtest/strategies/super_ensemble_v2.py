"""
Advanced Ensemble Strategy for XRPL Trading Bot.
Combines multiple strategies into a single orchestrator with:
1. Dynamic weighting based on recent performance (Adaptive Ensemble).
2. Risk management (Stop-Loss, Take-Profit) via Decorators.
3. Meta-Ensemble decision making.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import from local package structure
try:
    from backtest.strategies.base_strategy import (
        BaseStrategy, 
        Signal, 
        MetaEnsembleStrategy, 
        RiskManagedStrategy,
        EnsembleStrategy
    )
    from backtest.strategies.dynamic_ensemble import DynamicEnsembleStrategy
except ImportError:
    # Fallback for different execution environments
    from .base_strategy import (
        BaseStrategy, 
        Signal, 
        MetaEnsembleStrategy, 
        RiskManagedStrategy,
        EnsembleStrategy
    )
    from .dynamic_ensemble import DynamicEnsembleStrategy

logger = logging.getLogger(__name__)

class SuperEnsembleStrategy(BaseStrategy):
    """
    The ultimate ensemble orchestrator. 
    
    Architecture:
    - Input: A list of sub-strategies.
    - Layer 1 (Risk): Sub-strategies are wrapped in RiskManagedStrategy decorators.
    - Layer 2 (Dynamic Weighting): Wrapped strategies are fed into a DynamicEnsembleStrategy
      which adjusts weights based on recent implied performance.
    - Layer 3 (Meta Decision): The DynamicEnsemble provides the signal.
    """
    def __init__(
        self, 
        strategies: List[BaseStrategy], 
        use_risk_management: bool = True,
        risk_params: Optional[Dict[str, float]] = None,
        window_size: int = 20,
        adaptation_rate: float = 0.1
    ):
        super().__init__()
        self.use_risk_management = use_risk_management
        self.risk_params = risk_params or {'stop_loss_pct': 0.02, 'take_profit_pct': 0.05}
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # 1. Wrap strategies with Risk Management (Decorator Pattern)
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
        
        # 2. Initialize the Dynamic Ensemble with the processed strategies
        self.core_ensemble = DynamicEnsembleStrategy(
            strategies=processed_strategies,
            window_size=self.window_size,
            adaptation_rate=self.adaptation_rate
        )

    def init(self, data):
        self.data = data
        self.signals = []
        self.core_ensemble.init(data)

    def next(self, bar_idx: int) -> Signal:
        # The core ensemble manages the lifecycle and weighting
        return self.core_ensemble.next(bar_idx)

    def get_params(self) -> dict:
        params = {
            'use_risk_management': self.use_risk_management,
            'risk_params': self.risk_params,
            'window_size': self.window_size,
            'adaptation_rate': self.adaptation_rate
        }
        params.update(self.core_ensemble.get_params())
        return params
