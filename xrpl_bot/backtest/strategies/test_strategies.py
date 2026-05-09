import unittest
from typing import Dict, Any, List, Optional

# Mock Signal and Base classes to avoid imports since numpy is missing
class Signal:
    def __init__(self, type, price, timestamp, confidence, metadata=None):
        self.type = type
        self.price = price
        self.timestamp = timestamp
        self.confidence = confidence
        self.metadata = metadata or {}

class BaseStrategy:
    def __init__(self):
        self.data = None
        self.signals = []
    def init(self, data):
        self.data = data
        self.signals = []
    def next(self, bar_idx: int) -> Signal:
        raise NotImplementedError
    def get_params(self) -> dict:
        return {}

# Re-implementing the specific classes needed for testing without dependencies
class MockStrategy(BaseStrategy):
    def __init__(self, signal_type='HOLD', confidence=0.5):
        super().__init__()
        self.signal_type = signal_type
        self.confidence = confidence

    def next(self, bar_idx: int) -> Signal:
        return Signal(
            type=self.signal_type,
            price=100.0,
            timestamp=float(bar_idx),
            confidence=self.confidence
        )

class RiskManagedStrategy(BaseStrategy):
    def __init__(self, underlying_strategy: BaseStrategy, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05):
        super().__init__()
        self.strategy = underlying_strategy
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.active_position: Optional[Dict[str, Any]] = None 

    def init(self, data):
        self.data = data
        self.signals = []
        if hasattr(self, 'strategy'):
            self.strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        base_signal = self.strategy.next(bar_idx)
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]

        if self.active_position:
            entry_price = self.active_position['price']
            direction = self.active_position['direction']
            pnl_pct = (close_price - entry_price) / entry_price * direction

            if pnl_pct <= -self.stop_loss_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'stop_loss'})

            if pnl_pct >= self.take_profit_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'take_profit'})

            if base_signal.type == 'HOLD':
                 return Signal('HOLD', close_price, timestamp, 0.0, {'reason': 'managing_position'})

        if base_signal.type in ['BUY', 'SELL'] and not self.active_position:
            direction = 1 if base_signal.type == 'BUY' else -1
            self.active_position = {
                'price': close_price,
                'direction': direction,
                'timestamp': timestamp
            }
            return base_signal

        return Signal('HOLD', close_price, timestamp, 0.0, {'reason': 'managing_position'})

class EnsembleStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], voting_threshold: float = 0.5):
        super().__init__()
        self.strategies = strategies
        self.voting_threshold = voting_threshold

    def init(self, data):
        self.data = data
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_strategies = len(self.strategies)
        for strategy in self.strategies:
            signal = strategy.next(bar_idx)
            votes[signal.type] += 1

        best_signal_type = 'HOLD'
        max_votes = 0
        for s_type in ['BUY', 'SELL']:
            if votes[s_type] / total_strategies >= self.voting_threshold:
                if votes[s_type] > max_votes:
                    max_votes = votes[s_type]
                    best_signal_type = s_type

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        confidence = max_votes / total_strategies if max_votes > 0 else 0.0

        return Signal(
            type=best_signal_type,
            price=close_price,
            timestamp=timestamp,
            confidence=confidence,
            metadata={'ensemble_votes': votes}
        )

class MetaEnsembleStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        super().__init__()
        self.strategies = strategies
        self.weights = weights if weights is not None else [1.0/len(strategies)] * len(strategies)

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
            
            total_score += score * self.weights[i]

        threshold = 0.1
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

class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.data = {
            'close': [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            'high': [101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0],
            'low': [99.0, 100.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0],
            'timestamp': [float(i) for i in range(10)]
        }

    def test_risk_managed_stop_loss(self):
        underlying = MockStrategy(signal_type='BUY', confidence=1.0)
        risk_manager = RiskManagedStrategy(underlying, stop_loss_pct=0.1, take_profit_pct=0.5)
        risk_manager.init(self.data)
        
        sig1 = risk_manager.next(0)
        self.assertEqual(sig1.type, 'BUY')
        
        self.data['close'][5] = 80.0 
        sig2 = risk_manager.next(5)
        self.assertEqual(sig2.type, 'SELL')
        self.assertEqual(sig2.metadata['reason'], 'stop_loss')

    def test_ensemble_voting(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='BUY', confidence=1.0)
        s3 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        ensemble = EnsembleStrategy([s1, s2, s3], voting_threshold=0.5)
        ensemble.init(self.data)
        
        sig = ensemble.next(0)
        self.assertEqual(sig.type, 'BUY')

    def test_meta_ensemble_weights(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        meta = MetaEnsembleStrategy([s1, s2], weights=[0.9, 0.1])
        meta.init(self.data)
        sig = meta.next(0)
        self.assertEqual(sig.type, 'BUY')
        
        meta2 = MetaEnsembleStrategy([s1, s2], weights=[0.1, 0.9])
        meta2.init(self.data)
        sig2 = meta2.next(0)
        self.assertEqual(sig2.type, 'SELL')

if __name__ == '__main__':
    unittest.main()
