import numpy as np
from typing import List, Dict, Any, Optional
from xrpl_bot.backtest.strategies.base_strategy import BaseStrategy, Signal

class MetaEnsembleStrategy(BaseStrategy):
    """
    A weighted ensemble strategy where each sub-strategy's signal 
    is multiplied by a weight and summed.
    """
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        super().__init__()
        self.strategies = strategies
        if weights is not None:
            if len(weights) != len(strategies):
                raise ValueError("Weights must match number of strategies")
            # Ensure weights are a numpy array for vector math
            if hasattr(np, 'array'):
                self.weights = np.array(weights)
            else:
                self.weights = weights
        else:
            self.weights = [1.0/len(strategies)] * len(strategies)

    def init(self, data):
        self.data = data
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        total_score = 0.0
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]

        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            score = 0.0
            if signal.type == 'BUY':
                score = signal.confidence
            elif signal.type == 'SELL':
                score = -signal.confidence
            
            # Handle weights safely
            try:
                w = self.weights[i]
            except (TypeError, IndexError):
                w = 1.0 / len(self.strategies)
                
            total_score += score * w

        # Decision threshold
        threshold = 0.05
        if total_score > threshold:
            best_type = 'BUY'
        elif total_score < -threshold:
            best_type = 'SELL'
        else:
            best_type = 'HOLD'

        return Signal(
            type=best_type,
            price=close_price,
            timestamp=timestamp,
            confidence=abs(total_score),
            metadata={'meta_score': total_score}
        )

    def get_params(self) -> dict:
        # Check for tolist method (numpy) or fallback
        try:
            w_list = self.weights.tolist() if hasattr(self.weights, 'tolist') else self.weights
        except:
            w_list = self.weights
            
        params = {'weights': w_list}
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
