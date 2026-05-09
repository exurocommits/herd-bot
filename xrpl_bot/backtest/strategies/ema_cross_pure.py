"""
EMA Crossover Strategy — Pure Python, no numpy.
Fast EMA crosses above slow EMA -> BUY
Fast EMA crosses below slow EMA -> SELL
Confidence based on distance between EMAs relative to ATR.
"""
from .base_strategy import BaseStrategy, Signal
from .indicators import ema, atr, cross_above, cross_below


class EMACrossPure(BaseStrategy):
    """EMA Crossover with ATR-based confidence."""
    
    def __init__(self, fast_period: int = 9, slow_period: int = 21, atr_period: int = 14):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self._fast_ema = None
        self._slow_ema = None
        self._atr = None

    def init(self, data):
        self.data = data
        self.signals = []
        close = data['close']
        self._fast_ema = ema(close, self.fast_period)
        self._slow_ema = ema(close, self.slow_period)
        self._atr = atr(data['high'], data['low'], close, self.atr_period)

    def next(self, bar_idx: int) -> Signal:
        close = self.data['close']
        ts = self.data['timestamp']
        price = close[bar_idx]
        
        if bar_idx < self.slow_period:
            return Signal('HOLD', price, ts[bar_idx], 0.0)
        
        fast_val = self._fast_ema[bar_idx]
        slow_val = self._slow_ema[bar_idx]
        atr_val = self._atr[bar_idx]
        
        if any(v != v for v in [fast_val, slow_val, atr_val]):  # NaN check
            return Signal('HOLD', price, ts[bar_idx], 0.0)
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        if cross_above(self._fast_ema, self._slow_ema, bar_idx):
            signal_type = 'BUY'
            diff = abs(fast_val - slow_val)
            confidence = min(1.0, diff / atr_val) if atr_val > 0 else 0.5
        elif cross_below(self._fast_ema, self._slow_ema, bar_idx):
            signal_type = 'SELL'
            diff = abs(slow_val - fast_val)
            confidence = min(1.0, diff / atr_val) if atr_val > 0 else 0.5
        
        sig = Signal(signal_type, price, ts[bar_idx], confidence,
                     metadata={'fast_ema': fast_val, 'slow_ema': slow_val, 'atr': atr_val})
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {'fast_period': self.fast_period, 'slow_period': self.slow_period, 'atr_period': self.atr_period}
