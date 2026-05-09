import unittest
import numpy as np
from backtest.strategies.base_strategy import BaseStrategy, Signal, RiskManagedStrategy, EnsembleStrategy
from backtest.strategies.super_ensemble_v2 import SuperEnsembleStrategy

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

    def get_params(self) -> dict:
        return {'type': self.signal_type}

class TestEnsembleAndRisk(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.data = {
            'close': np.array([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 100.0] * 10),
            'high': np.array([102.0] * 100),
            'low': np.array([98.0] * 100),
            'timestamp': np.arange(100, dtype=float)
        }

    def test_risk_management_stop_loss(self):
        # Strategy always wants to BUY
        base = MockStrategy(signal_type='BUY', confidence=1.0)
        # Risk managed with 2% stop loss
        risk_managed = RiskManagedStrategy(base, stop_loss_pct=0.02, take_profit_pct=0.10)
        risk_managed.init(self.data)

        # Price goes from 100 to 97 (3% drop)
        # Bar 0: BUY at 100
        sig0 = risk_managed.next(0)
        self.assertEqual(sig0.type, 'BUY')

        # Bar 1: Price 101 (no stop)
        sig1 = risk_managed.next(1)
        self.assertEqual(sig1.type, 'HOLD')

        # Bar 2: Price 102 (no stop)
        sig2 = risk_managed.next(2)
        self.assertEqual(sig2.type, 'HOLD')

        # Manually force price down in data for testing
        self.data['close'][3] = 97.0 # 3% drop from 100
        sig3 = risk_managed.next(3)
        self.assertEqual(sig3.type, 'SELL')
        self.assertEqual(sig3.metadata['reason'], 'stop_loss')

    def test_super_ensemble_integration(self):
        s1 = MockStrategy(signal_type='BUY', confidence=0.8)
        s2 = MockStrategy(signal_type='BUY', confidence=0.8)
        
        ensemble = SuperEnsembleStrategy([s1, s2], use_risk_management=True)
        ensemble.init(self.data)
        
        signal = ensemble.next(1)
        self.assertIn(signal.type, ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(signal, Signal)

if __name__ == '__main__':
    unittest.main()
