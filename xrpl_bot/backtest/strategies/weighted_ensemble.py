from typing import List
import numpy as np
from backtest.strategies.base_strategy import BaseStrategy, Signal, RiskManagedStrategy

class WeightedEnsembleStrategy(BaseStrategy):
    """
    An ensemble strategy that combines multiple strategies using weighted voting.
    Weights can be assigned to each strategy based on their historical performance.
    """
    def __init__(self, strategies: List[BaseStrategy], weights: List[float] = None, voting_threshold: float = 0.5):
        super().__init__()
        self.strategies = strategies
        self.voting_threshold = voting_threshold
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            # Normalize weights
            sum_w = sum(weights)
            self.weights = [w / sum_w for w in weights]

    def init(self, data):
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        weighted_votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = sum(self.weights)

        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            weighted_votes[signal.type] += self.weights[i]

        # Determine the winner based on weighted votes
        best_signal_type = 'HOLD'
        max_weighted_vote = 0.0
        
        # We prioritize BUY/SELL over HOLD if they exceed the threshold
        for s_type in ['BUY', 'SELL']:
            if (weighted_votes[s_type] / total_weight) >= self.voting_threshold:
                if weighted_votes[s_type] > max_weighted_vote:
                    max_weighted_vote = weighted_votes[s_type]
                    best_signal_type = s_type

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        
        # Confidence is represented by the normalized weighted vote
        confidence = max_weighted_vote / total_weight if max_weighted_vote > 0 else 0.0

        return Signal(
            type=best_signal_type,
            price=close_price,
            timestamp=timestamp,
            confidence=confidence,
            metadata={'weighted_votes': weighted_votes}
        )

    def get_params(self) -> dict:
        params = {
            'voting_threshold': self.voting_threshold,
            'weights': self.weights
        }
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
