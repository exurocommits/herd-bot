"""
MACD Momentum Strategy — Pure Python, no numpy.
MACD line crosses above signal -> BUY
MACD line crosses below signal -> SELL
Uses histogram for confidence scoring.
"""
import math
from .base_strategy import BaseStrategy, Signal
from .indicators import macd, cross_above, cross_below


class MACDMomentumPure(BaseStrategy):
    """MACD crossover strategy with histogram-based confidence."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
        self._macd_line = None
        self._signal_line = None
        self._histogram = None

    def init(self, data):
        self.data = data
        self.signals = []
        close = data['close']
        self._macd_line, self._signal_line, self._histogram = macd(
            close, self.fast, self.slow, self.signal_period
        )

    def next(self, bar_idx: int) -> Signal:
        price = self.data['close'][bar_idx]
        ts = self.data['timestamp'][bar_idx]
        
        macd_val = self._macd_line[bar_idx]
        sig_val = self._signal_line[bar_idx]
        hist_val = self._histogram[bar_idx]
        
        if any(v != v for v in [macd_val, sig_val, hist_val]):
            return Signal('HOLD', price, ts, 0.0)
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        if cross_above(self._macd_line, self._signal_line, bar_idx):
            signal_type = 'BUY'
            # Confidence from histogram magnitude + zero-line position
            confidence = min(1.0, abs(hist_val) / (abs(macd_val) + 1e-9) * 2)
            if macd_val < 0:  # Below zero line = stronger buy
                confidence = min(1.0, confidence * 1.5)
        
        elif cross_below(self._macd_line, self._signal_line, bar_idx):
            signal_type = 'SELL'
            confidence = min(1.0, abs(hist_val) / (abs(macd_val) + 1e-9) * 2)
            if macd_val > 0:  # Above zero line = stronger sell
                confidence = min(1.0, confidence * 1.5)
        
        sig = Signal(signal_type, price, ts, confidence,
                     metadata={'macd': macd_val, 'signal': sig_val, 'histogram': hist_val})
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {'fast': self.fast, 'slow': self.slow, 'signal_period': self.signal_period}
