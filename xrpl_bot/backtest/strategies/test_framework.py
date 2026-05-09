import unittest
import sys
import os

# Add the strategies directory to sys.path to allow importing base_strategy
sys.path.append('/home/node/.openclaw/workspace/xrpl_bot/backtest/strategies')

from base_strategy import (
    Signal, 
    BaseStrategy, 
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
        # Mock data access
        close = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        return Signal(self.signal_type, close, timestamp, self.confidence)

class TestStrategyFramework(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.data = {
            'close': [100.0, 105.0, 110.0, 105.0, 100.0, 95.0, 100.0],
            'high': [102.0, 107.0, 112.0, 107.0, 102.0, 97.0, 102.0],
            'low': [98.0, 103.0, 108.0, 103.0, 98.0, 93.0, 98.0],
            'timestamp': [float(i) for i in range(7)]
        }

    def test_risk_managed_strategy_stop_loss(self):
        # Strategy that always buys
        underlying = MockStrategy(signal_type='BUY', confidence=1.0)
        # 10% stop loss
        risk_manager = RiskManagedStrategy(underlying, stop_loss_pct=0.10, take_profit_pct=0.50)
        risk_manager.init(self.data)

        # Bar 0: Entry (Price 100)
        sig0 = risk_manager.next(0)
        self.assertEqual(sig0.type, 'BUY')
        self.assertEqual(risk_manager.active_position['price'], 100.0)

        # Bar 1: Price 105 (In profit, no SL/TP)
        sig1 = risk_manager.next(1)
        self.assertEqual(sig1.type, 'HOLD')

        # Bar 2: Price 110 (In profit)
        sig2 = risk_manager.next(2)
        self.assertEqual(sig2.type, 'HOLD')

        # Bar 3: Price 105 (Still in profit, 105 > 100*0.9)
        sig3 = risk_manager.next(3)
        self.assertEqual(sig3.type, 'HOLD')

        # Bar 4: Price 100 (Break even)
        sig4 = risk_manager.next(4)
        self.assertEqual(sig4.type, 'HOLD')

        # Bar 5: Price 85 (Force a drop to trigger SL)
        # We need to inject 85 into data for this test to work properly
        self.data['close'][5] = 85.0
        sig5 = risk_manager.next(5)
        self.assertEqual(sig5.type, 'SELL')
        self.assertEqual(sig5.metadata['reason'], 'stop_loss')
        self.assertIsNone(risk_manager.active_position)

    def test_ensemble_voting(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='BUY', confidence=1.0)
        s3 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        # Threshold 0.6: 2/3 = 0.66 -> BUY
        ensemble = EnsembleStrategy([s1, s2, s3], voting_threshold=0.6)
        ensemble.init(self.data)
        
        sig = ensemble.next(0)
        self.assertEqual(sig.type, 'BUY')
        self.assertEqual(sig.metadata['ensemble_votes']['BUY'], 2)

    def test_meta_ensemble_weighted(self):
        s1 = MockStrategy(signal_type='BUY', confidence=1.0) # Weight 0.8
        s2 = MockStrategy(signal_type='SELL', confidence=1.0) # Weight 0.2
        
        # Score = (1.0 * 0.8) + (-1.0 * 0.2) = 0.6
        # 0.6 > threshold (0.1) -> BUY
        meta = MetaEnsembleStrategy([s1, s2], weights=[0.8, 0.2])
        meta.init(self.data)
        
        sig = meta.next(0)
        self.assertEqual(sig.type, 'BUY')
        self.assertAlmostEqual(sig.confidence, 0.6)

if __name__ == '__main__':
    unittest.main()
