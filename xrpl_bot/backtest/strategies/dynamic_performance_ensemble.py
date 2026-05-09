import numpy as np
from typing import List, Dict, Any, Optional
from xrpl_bot.backtest.strategies.base_strategy import BaseStrategy, Signal

class DynamicPerformanceEnsemble(BaseStrategy):
    """
    An ensemble strategy that adjusts weights of sub-strategies 
    based on their recent performance (Sharpe-like score).
    """
    def __init__(self, strategies: List[BaseStrategy], lookback_window: int = 20, learning_rate: float = 0.1):
        super().__init__()
        self.strategies = strategies
        self.lookback_window = lookback_window
        self.learning_rate = learning_rate
        self.weights = np.array([1.0 / len(strategies)] * len(strategies))
        # Track recent returns for each strategy to update weights
        # stores list of (return_pct) for each strategy
        self.performance_history: List[List[float]] = [[] for _ in range(len(strategies))]
        self.last_prices = {} # strategy_idx -> last_price

    def init(self, data):
        self.data = data
        self.signals = []
        for i, strategy in enumerate(self.strategies):
            strategy.init(data)
            self.last_prices[i] = self.data['close'][0]
        self.weights = np.array([1.0 / len(self.strategies)] * len(self.strategies))
        self.performance_history = [[] for _ in range(len(self.strategies))]

    def _update_weights(self, bar_idx: int):
        """
        Adjust weights based on the performance of strategies in the lookback window.
        """
        for i in range(len(self.strategies)):
            if len(self.performance_history[i]) > 0:
                # Use mean return as a simple performance proxy
                avg_return = sum(self.performance_history[i][-self.lookback_window:]) / min(len(self.performance_history[i]), self.lookback_window)
                # Update weight: move towards higher returns
                # This is a very naive gradient-descent-like update
                self.weights[i] *= (1.0 + self.learning_rate * avg_return)
        
        # Re-normalize weights
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            self.weights = np.array([1.0 / len(self.strategies)] * len(self.strategies))

    def next(self, bar_idx: int) -> Signal:
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        
        total_score = 0.0
        combined_signal = 'HOLD'

        # 1. Get signals and track performance
        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            
            # Track performance (return from previous bar)
            prev_price = self.last_prices[i]
            price_return = (close_price - prev_price) / prev_price
            
            # If the strategy was signaling BUY or SELL last bar, we attribute the return
            if hasattr(strategy, 'last_signal_type') and strategy.last_signal_type == 'BUY':
                self.performance_history[i].append(price_return)
            elif hasattr(strategy, 'last_signal_type') and strategy.last_signal_type == 'SELL':
                self.performance_history[i].append(-price_return)
            else:
                self.performance_history[i].append(0.0)

            self.last_prices[i] = close_price
            
            # Calculate score for ensemble
            score = 0.0
            if signal.type == 'BUY':
                score = signal.confidence
            elif signal.type == 'SELL':
                score = -signal.confidence
            
            total_score += score * self.weights[i]
            
            # Store for next bar
            strategy.last_signal_type = signal.type

        # 2. Update weights periodically
        if bar_idx % 5 == 0:
            self._update_weights(bar_idx)

        # 3. Decision
        threshold = 0.05
        if total_score > threshold:
            combined_signal = 'BUY'
        elif total_score < -threshold:
            combined_signal = 'SELL'
        
        return Signal(
            type=combined_signal,
            price=close_price,
            timestamp=timestamp,
            confidence=abs(total_score),
            metadata={'weights': self.weights.tolist(), 'meta_score': total_score}
        )

    def get_params(self) -> dict:
        params = {
            'lookback_window': self.lookback_window,
            'learning_rate': self.learning_rate,
            'weights': self.weights.tolist()
        }
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
