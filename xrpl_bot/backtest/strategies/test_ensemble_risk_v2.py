from typing import List, Dict, Any
import unittest

# Manual Mocking of Numpy and Signal since we can't install it
class MockNumpy:
    @staticmethod
    def array(x, *args, **kwargs): return x
    @staticmethod
    def argmax(x): return 0
    @staticmethod
    def percentile(x, q): return 0

class Signal:
    def __init__(self, type, price, timestamp, confidence, metadata=None):
        self.type = type
        self.price = price
        self.timestamp = timestamp
        self.confidence = confidence
        self.metadata = metadata or {}

from xrpl_bot.backtest.strategies.base_strategy import (
    BaseStrategy, 
    RiskManagedStrategy, 
    EnsembleStrategy, 
    MetaEnsembleStrategy
)

class MockStrategy(BaseStrategy):
    def __init__(self, signal_type='BUY', confidence=1.0):
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

class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Create dummy data without numpy dependency in the dict values
        self.data = {
            'close': [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 99.0],
            'high': [102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0],
            'low': [98.0, 99.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0],
            'timestamp': [float(i) for i in range(10)]
        }

    def test_ensemble_voting(self):
        s1 = MockStrategy('BUY')
        s2 = MockStrategy('BUY')
        s3 = MockStrategy('SELL')
        
        ensemble = EnsembleStrategy([s1, s2, s3], voting_threshold=0.5)
        ensemble.init(self.data)
        signal = ensemble.next(5)
        self.assertEqual(signal.type, 'BUY')
        self.assertGreater(signal.confidence, 0.5)

    def test_meta_ensemble_weighted(self):
        s1 = MockStrategy('BUY', confidence=1.0)
        s2 = MockStrategy('SELL', confidence=1.0)
        
        meta = MetaEnsembleStrategy([s1, s2], weights=[0.8, 0.2])
        meta.init(self.data)
        signal = meta.next(5)
        self.assertEqual(signal.type, 'BUY')

    def test_risk_management_stop_loss(self):
        underlying = MockStrategy('BUY')
        risk_managed = RiskManagedStrategy(underlying, stop_loss_pct=0.05, take_profit_pct=0.20)
        risk_managed.init(self.data)

        # 1. Entry
        sig_entry = risk_managed.next(0)
        self.assertEqual(sig_entry.type, 'BUY')
        self.assertIsNotNone(risk_managed.active_position)

        # 2. Force a stop loss by manually setting active_position price
        risk_managed.active_position['price'] = 105.0
        # Current price at index 7 is 97.0. (97-105)/105 = -7.6%
        sig_exit = risk_managed.next(7)
        self.assertEqual(sig_exit.type, 'SELL')
        self.assertEqual(sig_exit.metadata['reason'], 'stop_loss')
        self.assertIsNone(risk_managed.active_position)

if __name__ == '__main__':
    unittest.main()
