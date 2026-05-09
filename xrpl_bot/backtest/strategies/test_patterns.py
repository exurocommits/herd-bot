import sys
import os
import unittest
from typing import List

# Ensure we can import the strategies
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from backtest.strategies.base_strategy import (
        BaseStrategy, 
        Signal, 
        RiskManagedStrategy, 
        EnsembleStrategy, 
        MetaEnsembleStrategy
    )
except ImportError:
    # Fallback for running directly if PYTHONPATH is set correctly
    from strategies.base_strategy import (
        BaseStrategy, 
        Signal, 
        RiskManagedStrategy, 
        EnsembleStrategy, 
        MetaEnsembleStrategy
    )

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

class TestStrategyPatterns(unittest.TestCase):
    def setUp(self):
        self.data = {
            'close': [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 100.0],
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'timestamp': [float(i) for i in range(10)]
        }

    def test_risk_managed_stop_loss(self):
        # Strategy that always says BUY
        base = MockStrategy(signal_type='BUY', confidence=1.0)
        # 10% stop loss
        risk_managed = RiskManagedStrategy(base, stop_loss_pct=0.1, take_profit_pct=0.5)
        risk_managed.init(self.data)

        # First bar: Should trigger BUY
        sig1 = risk_managed.next(0)
        self.assertEqual(sig1.type, 'BUY')
        self.assertIsNotNone(risk_managed.active_position)
        entry_price = risk_managed.active_position['price']

        # Simulate a massive drop in the next bar to trigger stop loss
        # We need to modify the data for the next call
        self.data['close'][1] = entry_price * 0.8 # 20% drop
        
        sig2 = risk_managed.next(1)
        self.assertEqual(sig2.type, 'SELL')
        self.assertEqual(sig2.metadata['reason'], 'stop_loss')
        self.assertIsNone(risk_managed.active_position)

    def test_ensemble_voting(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='BUY', confidence=1.0)
        s3 = MockStrategy(signal_type='HOLD', confidence=1.0)
        
        # Threshold 0.5. 2/3 = 0.66 -> BUY
        ensemble = EnsembleStrategy([s1, s2, s3], voting_threshold=0.5)
        ensemble.init(self.data)
        
        sig = ensemble.next(0)
        self.assertEqual(sig.type, 'BUY')
        self.assertEqual(sig.metadata['ensemble_votes']['BUY'], 2)

    def test_meta_ensemble_weighted(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        # s1 weight 0.8, s2 weight 0.2
        # Score = (1.0 * 0.8) + (-1.0 * 0.2) = 0.6 -> BUY
        meta = MetaEnsembleStrategy([s1, s2], weights=[0.8, 0.2])
        meta.init(self.data)
        
        sig = meta.next(0)
        self.assertEqual(sig.type, 'BUY')
        self.assertAlmostEqual(sig.confidence, 0.6)

if __name__ == '__main__':
    unittest.main()
