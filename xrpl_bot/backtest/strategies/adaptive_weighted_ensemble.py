from typing import List
import numpy as np
from backtest.strategies.base_strategy import BaseStrategy, Signal, RiskManagedStrategy

class AdaptiveWeightedEnsembleStrategy(BaseStrategy):
    """
    An ensemble strategy that combines multiple strategies using adaptive weights.
    Weights are adjusted based on the recent success (direction accuracy) of each strategy.
    """
    def __init__(self, strategies: List[BaseStrategy], window_size: int = 20, learning_rate: float = 0.1, voting_threshold: float = 0.5):
        super().__init__()
        self.strategies = strategies
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.voting_threshold = voting_threshold
        
        # Initialize equal weights
        num_strats = len(strategies)
        self.weights = np.ones(num_strats) / num_strats
        
        # Track recent performance per strategy
        # stores (signal_type, direction_of_next_price_move)
        self.performance_history = [[] for _ in range(num_strats)]

    def init(self, data):
        for strategy in self.strategies:
            strategy.init(data)
        self.data = data
        self._prev_signals = [None] * len(self.strategies)

    def next(self, bar_idx: int) -> Signal:
        # Update weights based on the signal from bar_idx - 1
        if bar_idx > 1 and any(s is not None for s in self._prev_signals):
            price_prev = self.data['close'][bar_idx - 1]
            price_curr = self.data['close'][bar_idx]
            actual_move = 1 if price_curr > price_prev else (-1 if price_curr < price_prev else 0)

            if actual_move != 0:
                for i in range(len(self.strategies)):
                    prev_sig = self._prev_signals[i]
                    if prev_sig and prev_sig.type in ['BUY', 'SELL']:
                        pred_move = 1 if prev_sig.type == 'BUY' else -1
                        accuracy = 1.0 if pred_move == actual_move else -1.0
                        self.weights[i] += self.learning_rate * accuracy
                        self.weights[i] = max(0.001, self.weights[i])
                
                sum_w = np.sum(self.weights)
                self.weights /= sum_w

        # 2. Generate current signals
        weighted_votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            self._prev_signals[i] = signal 
            weighted_votes[signal.type] += self.weights[i]

        # 3. Determine the winner
        best_signal_type = 'HOLD'
        max_weighted_vote = 0.0
        
        sum_weights = np.sum(self.weights)
        for s_type in ['BUY', 'SELL']:
            if (weighted_votes[s_type] / sum_weights) >= self.voting_threshold:
                if weighted_votes[s_type] > max_weighted_vote:
                    max_weighted_vote = weighted_votes[s_type]
                    best_signal_type = s_type

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        confidence = max_weighted_vote / sum_weights if max_weighted_vote > 0 else 0.0

        return Signal(
            type=best_signal_type,
            price=close_price,
            timestamp=timestamp,
            confidence=confidence,
            metadata={
                'weighted_votes': weighted_votes,
                'weights': self.weights.tolist()
            }
        )

    def get_params(self) -> dict:
        params = {
            'voting_threshold': self.voting_threshold,
            'learning_rate': self.learning_rate,
            'window_size': self.window_size
        }
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
