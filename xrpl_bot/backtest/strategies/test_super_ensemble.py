"""
Unit tests for the new SuperEnsembleStrategy and its components.
Ensures that the ensemble properly delegates to sub-strategies and 
that risk management (via RiskManagedStrategy) is correctly applied.
"""
import unittest
from dataclasses import dataclass
from backtest.strategies.base_strategy import (
    BaseStrategy, 
    Signal, 
    RiskManagedStrategy, 
    EnsembleStrategy
)
from backtest.strategies.super_ensemble import SuperEnsembleStrategy

class MockStrategy(BaseStrategy):
    """A simple strategy that returns a constant signal."""
    def __init__(self, signal_type='HOLD', confidence=1.0):
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

class TestSuperEnsemble(unittest.TestCase):
    def setUp(self):
        # Mock data
        self.data = {
            'close': [100.0, 105.0, 102.0, 98.0, 110.0, 105.0, 100.0, 95.0, 110.0, 120.0],
            'high': [105.0, 110.0, 105.0, 100.0, 115.0, 110.0, 105.0, 100.0, 115.0, 125.0],
            'low': [95.0, 100.0, 97.0, 93.0, 105.0, 100.0, 95.0, 90.0, 105.0, 115.0],
            'timestamp': [float(i) for i in range(10)]
        }

    def test_ensemble_delegation(self):
        """Test if SuperEnsemble correctly returns signals from its sub-strategies."""
        s1 = MockStrategy(signal_type='BUY')
        s2 = MockStrategy(signal_type='HOLD')
        
        # Using 2 strategies, threshold should be met for BUY if s1 is dominant
        ensemble = SuperEnsembleStrategy(
            strategies=[s1, s2], 
            use_risk_management=False
        )
        ensemble.init(self.data)
        
        signal = ensemble.next(1)
        self.assertEqual(signal.type, 'BUY')
        self.assertGreater(signal.confidence, 0.0)

    def test_risk_management_integration(self):
        """Test if RiskManagedStrategy (inside SuperEnsemble) triggers a stop loss."""
        # Strategy that stays 'BUY' forever
        s1 = MockStrategy(signal_type='BUY')
        
        # Risk params: 2% SL, 10% TP
        ensemble = SuperEnsembleStrategy(
            strategies=[s1],
            use_risk_management=True,
            risk_params={'stop_loss_pct': 0.02, 'take_profit_pct': 0.10}
        )
        ensemble.init(self.data)
        
        # Step 1: Enter position at bar 0 (price 100)
        signal_entry = ensemble.next(0)
        self.assertEqual(signal_entry.type, 'BUY')
        
        # Step 2: Price drops to 97 (3% drop) at bar 3 (data['close'][3] = 98, data['close'][7] = 95)
        # Let's use bar 7 where price is 95.0
        signal_sl = ensemble.next(7)
        self.assertEqual(signal_sl.type, 'SELL')
        self.assertEqual(signal_sl.metadata.get('reason'), 'stop_loss')

    def test_dynamic_weighting_logic(self):
        """Test that recording PnL affects the ensemble (smoke test)."""
        s1 = MockStrategy(signal_type='BUY')
        s2 = MockStrategy(signal_type='SELL')
        
        ensemble = SuperEnsembleStrategy(
            strategies=[s1, s2],
            use_risk_management=False,
            lookback_period=5
        )
        ensemble.init(self.data)
        
        # Record positive PnL for s1 and negative for s2
        ensemble.record_pnl(0, 10.0)
        ensemble.record_pnl(1, 10.0)
        ensemble.record_pnl(1, -10.0)
        
        # Trigger weight update (every 5 bars)
        # bar_idx 5 should trigger update
        signal = ensemble.next(5)
        
        # Since s1 was profitable, it should have much higher weight
        # The ensemble signal should reflect s1's direction (BUY)
        self.assertEqual(signal.type, 'BUY')

if __name__ == '__main__':
    unittest.main()
