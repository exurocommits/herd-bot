"""
Supertrend Follower Strategy — Pure Python, no numpy.
Follows the Supertrend indicator for trend-following signals.
BUY when price closes above Supertrend (direction flips up)
SELL when price closes below Supertrend (direction flips down)
Confidence: distance from Supertrend / ATR.
"""
import math
from .base_strategy import BaseStrategy, Signal
from .indicators import supertrend


class SupertrendPure(BaseStrategy):
    """Supertrend trend-following strategy."""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__()
        self.period = period
        self.multiplier = multiplier
        self._st = None
        self._direction = None
        self._prev_direction = 0

    def init(self, data):
        self.data = data
        self.signals = []
        self._st, self._direction = supertrend(
            data['high'], data['low'], data['close'],
            self.period, self.multiplier
        )
        self._prev_direction = 0

    def next(self, bar_idx: int) -> Signal:
        price = self.data['close'][bar_idx]
        ts = self.data['timestamp'][bar_idx]
        
        st_val = self._st[bar_idx]
        curr_dir = self._direction[bar_idx]
        
        if math.isnan(st_val) or curr_dir == 0:
            self._prev_direction = curr_dir
            return Signal('HOLD', price, ts, 0.0)
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # Direction flip detection
        if curr_dir == 1 and self._prev_direction == -1:
            # Flipped from down to up -> BUY
            signal_type = 'BUY'
            dist = abs(price - st_val)
            atr_approx = st_val * 0.02  # rough ATR proxy for confidence
            confidence = min(1.0, dist / atr_approx) if atr_approx > 0 else 0.6
        
        elif curr_dir == -1 and self._prev_direction == 1:
            # Flipped from up to down -> SELL
            signal_type = 'SELL'
            dist = abs(price - st_val)
            atr_approx = st_val * 0.02
            confidence = min(1.0, dist / atr_approx) if atr_approx > 0 else 0.6
        
        self._prev_direction = curr_dir
        
        sig = Signal(signal_type, price, ts, confidence,
                     metadata={'supertrend': st_val, 'direction': curr_dir})
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {'period': self.period, 'multiplier': self.multiplier}
