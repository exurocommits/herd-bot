from typing import List, Dict, Any
import numpy as np
from .base_strategy import BaseStrategy, Signal, EnsembleStrategy, RiskManagedStrategy

class MetaStrategy(BaseStrategy):
    """
    An advanced ensemble strategy that uses weighted voting based on 
    historical performance (simulated) or signal confidence.
    """
    def __init__(self, strategies: List[BaseStrategy], weights: List[float] = None):
        super().__init__()
        self.strategies = strategies
        if weights:
            if len(weights) != len(strategies):
                raise ValueError("Weights length must match strategies length")
            self.weights = np.array(weights)
        else:
            self.weights = np.ones(len(strategies)) / len(strategies)

    def init(self, data):
        super().init(data)
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        buy_score = 0.0
        sell_score = 0.0
        
        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            weight = self.weights[i]
            
            if signal.type == 'BUY':
                buy_score += weight * signal.confidence
            elif signal.type == 'SELL':
                sell_score += weight * signal.confidence

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        
        best_type = 'HOLD'
        confidence = 0.0
        
        if buy_score > sell_score and buy_score > 0.5:
            best_type = 'BUY'
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            best_type = 'SELL'
            confidence = sell_score

        return Signal(
            type=best_type,
            price=close_price,
            timestamp=timestamp,
            confidence=confidence,
            metadata={'buy_score': buy_score, 'sell_score': sell_score}
        )

class DynamicEnsemble(BaseStrategy):
    """
    An ensemble that adjusts weights based on recent signal accuracy.
    (Stub for future implementation)
    """
    def __init__(self, strategies: List[BaseStrategy]):
        super().__init__()
        self.strategies = strategies
        self.performance_registry = {id(s): 1.0 for s in strategies}

    def init(self, data):
        super().init(data)
        for s in self.strategies:
            s.init(data)

    def next(self, bar_idx: int) -> Signal:
        # Simple implementation for now: fallback to standard Ensemble
        return EnsembleStrategy(self.strategies).next(bar_idx)
