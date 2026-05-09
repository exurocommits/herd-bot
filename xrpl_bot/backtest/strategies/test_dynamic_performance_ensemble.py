import unittest
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Fallback for environments without numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    class MockNumpy:
        def __init__(self): pass
        def array(self, x, *args, **kwargs): return x
        def mean(self, x): return sum(x)/len(x) if len(x)>0 else 0
        def zeros(self, n): return [0.0]*n
        def isna(self, x): return [False]*len(x)
        def nan(self): return float('nan')
        def abs(self, x): return [abs(i) for i in x]
        def percentile(self, q): return 0
        def argmax(self, x): return 0
        def __getitem__(self, key): return self
        def __call__(self, *args, **kwargs): return self
        def tolist(self): return []
    np = MockNumpy


def np_abs(x):
    if HAS_NUMPY:
        return np.abs(x)
    return [abs(i) for i in x]

from backtest.strategies.base_strategy import BaseStrategy, Signal
from backtest.strategies.dynamic_performance_ensemble import DynamicEnsembleStrategy

class MockStrategy(BaseStrategy):
    """A strategy that returns a fixed signal for testing."""
    def __init__(self, signal_type='HOLD', confidence=1.0):
        super().__init__()
        self.signal_type = signal_type
        self.confidence = confidence
        self.data = None

    def init(self, data):
        self.data = data

    def next(self, bar_idx: int) -> Signal:
        return Signal(
            type=self.signal_type,
            price=float(self.data['close'][bar_idx]),
            timestamp=float(self.data['timestamp'][bar_idx]),
            confidence=self.confidence
        )

    def get_params(self) -> dict:
        return {"type": self.signal_type}

class TestDynamicEnsemble(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.data = {
            'close': np.array([100.0, 101.0, 102.0, 101.0, 100.0, 102.0, 103.0, 104.0, 105.0, 106.0]),
            'timestamp': np.arange(10, dtype=float)
        }

    def test_equal_weights_init(self):
        s1 = MockStrategy(signal_type='BUY')
        s2 = MockStrategy(signal_type='SELL')
        ensemble = DynamicEnsembleStrategy([s1, s2])
        ensemble.init(self.data)
        
        # Initial weights should be equal
        self.assertEqual(len(ensemble.weights), 2)
        self.assertAlmostEqual(ensemble.weights[0], 0.5)
        self.assertAlmostEqual(ensemble.weights[1], 0.5)

    def test_weight_update_on_pnl(self):
        s1 = MockStrategy(signal_type='BUY')
        s2 = MockStrategy(signal_type='SELL')
        ensemble = DynamicEnsembleStrategy([s1, s2], lookback_period=5, smoothing=1.0)
        ensemble.init(self.data)

        # Feed PnL: s1 is great, s2 is terrible
        ensemble.record_strategy_pnl(0, 10.0)
        ensemble.record_strategy_pnl(1, -10.0)

        # Trigger weight update (every 5 bars in DynamicEnsembleStrategy.next)
        # We'll simulate bar_idx = 5
        ensemble.next(5)
        
        # Weight for s1 (the winner) should be significantly higher than s2
        self.assertGreater(ensemble.weights[0], ensemble.weights[1])
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)

    def test_voting_logic(self):
        # Two BUY strategies, one SELL strategy
        s1 = MockStrategy(signal_type='BUY', confidence=1.0)
        s2 = MockStrategy(signal_type='BUY', confidence=1.0)
        s3 = MockStrategy(signal_type='SELL', confidence=1.0)
        
        ensemble = DynamicEnsembleStrategy([s1, s2, s3])
        ensemble.init(self.data)
        
        # With equal weights (1/3 each):
        # BUY weight = 1/3 + 1/3 = 2/3
        # SELL weight = 1/3
        # Result should be BUY
        signal = ensemble.next(1)
        self.assertEqual(signal.type, 'BUY')
        self.assertGreater(signal.confidence, 0.5)

    def test_empty_history_fallback(self):
        s1 = MockStrategy(signal_type='BUY')
        s2 = MockStrategy(signal_type='SELL')
        ensemble = DynamicEnsembleStrategy([s1, s2])
        ensemble.init(self.data)
        
        # Try to update weights without any history
        ensemble._update_weights(1)
        self.assertEqual(ensemble.weights, [0.5, 0.5])

if __name__ == '__main__':
    unittest.main()
