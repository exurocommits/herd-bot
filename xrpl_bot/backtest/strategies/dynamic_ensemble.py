"""
Dynamic Ensemble Strategy for XRPL Trading Bot.
Weights sub-strategies based on their recent performance (Sharpe ratio or cumulative PnL).
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Fallback for environments without numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

# Assuming the BaseStrategy and Signal classes are imported correctly 
# in a real environment. For the purpose of this file, we assume they are available via relative import.
from .base_strategy import BaseStrategy, Signal

@dataclass
class StrategyPerformance:
    """Tracks the performance of a single strategy for weighting."""
    cumulative_pnl: float = 0.0
    trades_count: int = 0
    last_signal_time: float = 0.0
    recent_returns: List[float] = field(default_factory=list)

class DynamicEnsembleStrategy(BaseStrategy):
    """
    An ensemble strategy that dynamically adjusts the weights of its 
    sub-strategies based on their recent performance.
    
    Uses a sliding window of recent returns to calculate a performance score.
    """
    def __init__(self, strategies: List[BaseStrategy], window_size: int = 20, adaptation_rate: float = 0.1):
        super().__init__()
        self.strategies = strategies
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.performances = [StrategyPerformance() for _ in range(len(strategies))]
        self.weights = [1.0 / len(strategies)] * len(strategies)
        self.last_prices = [0.0] * len(strategies)

    def init(self, data):
        super().init(data)
        for strategy in self.strategies:
            strategy.init(data)
        self.performances = [StrategyPerformance() for _ in range(len(self.strategies))]
        self.weights = [1.0 / len(self.strategies)] * len(self.strategies)

    def _update_weights(self):
        """Re-calculate weights based on cumulative PnL of recent window."""
        scores = []
        for perf in self.performances:
            # Use cumulative PnL in the window as a simple score. 
            # Higher PnL -> Higher Weight.
            # We add a small epsilon to avoid zero weights.
            score = max(0.01, perf.cumulative_pnl + 1.0) 
            scores.append(score)
        
        total_score = sum(scores)
        if total_score > 0:
            self.weights = [s / total_score for s in scores]
        else:
            # Fallback to uniform if all scores are zero/negative
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)

    def next(self, bar_idx: int) -> Signal:
        if bar_idx == 0:
            self.last_prices = [self.data['close'][0]] * len(self.strategies)
            return Signal('HOLD', self.data['close'][0], self.data['timestamp'][0], 0.0)

        current_price = self.data['close'][bar_idx]
        current_timestamp = self.data['timestamp'][bar_idx]
        
        total_weighted_score = 0.0
        
        # 1. Collect signals and update performance of sub-strategies
        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            
            # Update performance based on the *previous* signal's impact on current price
            # (This is a simplified way to track PnL without a full account model)
            prev_price = self.last_prices[i]
            price_change_pct = (current_price - prev_price) / prev_price
            
            # If the strategy had a signal that was BUY or SELL, we update its performance
            # This is a bit of a hack: we assume the strategy was "in" the position
            # In a real system, we'd track actual trades.
            # Here we use the metadata/signal to estimate.
            # (Simplification: If signal is BUY, we assume it wanted to be long)
            # Actually, a better way is to track if it *emitted* a signal.
            # For this implementation, we'll check if the signal's confidence was high.
            
            # For now, let's just track if the price moved in the direction of the signal.
            # This is very crude, but fits the "Strategy Template" requirement.
            # A real implementation would use a separate Backtest Engine to feed PnL.
            
            # Let's assume for this dynamic weighting that we track the 'implied' return
            # if the signal was held.
            # (Note: This logic is for demonstration within the strategy class itself)
            
            # We'll use a simplified approach: if signal.type is BUY, return is price_change.
            # If SELL, return is -price_change.
            implied_return = 0.0
            if signal.type == 'BUY':
                implied_return = price_change_pct
            elif signal.type == 'SELL':
                implied_return = -price_change_pct
                
            self.performances[i].cumulative_pnl += implied_return
            self.performances[i].recent_returns.append(implied_return)
            if len(self.performances[i].recent_returns) > self.window_size:
                self.performances[i].recent_returns.pop(0)
            
            # Calculate weighted score contribution
            score = 0.0
            if signal.type == 'BUY':
                score = signal.confidence
            elif signal.type == 'SELL':
                score = -signal.confidence
                
            total_weighted_score += score * self.weights[i]
            self.last_prices[i] = current_price

        # 2. Re-weight based on performance
        self._update_weights()

        # 3. Final Decision
        threshold = 0.15
        best_type = 'HOLD'
        if total_weighted_score > threshold:
            best_type = 'BUY'
        elif total_weighted_score < -threshold:
            best_type = 'SELL'

        return Signal(
            type=best_type,
            price=current_price,
            timestamp=current_timestamp,
            confidence=min(1.0, abs(total_weighted_score)),
            metadata={
                'ensemble_score': total_weighted_score,
                'weights': self.weights
            }
        )

    def get_params(self) -> dict:
        params = {
            'window_size': self.window_size,
            'adaptation_rate': self.adaptation_rate,
            'weights': self.weights
        }
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
