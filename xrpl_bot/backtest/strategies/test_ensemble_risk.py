import unittest
import numpy as np
from xrpl_bot.backtest.strategies.base_strategy import BaseStrategy, Signal, RiskManagedStrategy, EnsembleStrategy, MetaEnsembleStrategy

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

class TestEnsembleAndRisk(unittest.TestCase):
    def setUp(self):
        self.data = {
            'close': np.array([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0]),
            'high': np.array([101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0]),
            'low': np.array([99.0, 100.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0]),
            'timestamp': np.array([float(i) for i in range(10)])
        }

    def test_risk_managed_stop_loss(self):
        # Strategy that always buys
        base = MockStrategy(signal_type='BUY', confidence=1.0)
        # 5% Stop Loss
        risk_managed = RiskManagedStrategy(base, stop_loss_pct=0.05, take_profit_pct=0.5)
        risk_managed.init(self.data)

        # Bar 0: Buy at 100
        sig0 = risk_managed.next(0)
        self.assertEqual(sig0.type, 'BUY')

        # Bar 1: Price 101 (no exit)
        sig1 = risk_managed.next(1)
        self.assertEqual(sig1.type, 'HOLD')

        # Bar 5: Price 99.0. (100 -> 99 is -1%. Not hit)
        sig5 = risk_managed.next(5)
        self.assertEqual(sig5.type, 'HOLD')

        # Bar 9: Price 95.0. (100 -> 95 is -5%. Hit!)
        # Note: the data index 9 is 95.0.
        sig9 = risk_managed.next(9)
        self.assertEqual(sig9.type, 'SELL')
        self.assertEqual(sig9.metadata['reason'], 'stop_loss')

    def test_ensemble_voting(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='BUY', confidence=1.0)
        s3 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        # Threshold 0.5. 2/3 = 0.66 > 0.5. Should be BUY.
        ensemble = EnsembleStrategy([s1, s2, s3], voting_threshold=0.5)
        ensemble.init(self.data)
        
        sig = ensemble.next(1)
        self.assertEqual(sig.type, 'BUY')
        self.assertIn('ensemble_votes', sig.metadata)

    def test_meta_ensemble_weighted(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        # Weight s1=0.8, s2=0.2. 
        # Score = (1.0 * 0.8) + (-1.0 * 0.2) = 0.6.
        # 0.6 > 0.05 threshold -> BUY
        meta = MetaEnsembleStrategy([s1, s2], weights=[0.8, 0.2])
        meta.init(self.data)
        
        sig = meta.next(1)
        self.assertEqual(sig.type, 'BUY')

if __name__ == '__main__':
    unittest.main()
