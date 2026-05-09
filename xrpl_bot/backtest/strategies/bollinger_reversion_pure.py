"""
Bollinger Band Mean Reversion Strategy — Pure Python, no numpy.
Price touches lower band + RSI oversold -> BUY
Price touches upper band + RSI overbought -> SELL
Confidence: distance beyond band edge.
"""
import math
from .base_strategy import BaseStrategy, Signal
from .indicators import bollinger_bands, rsi


class BollingerReversionPure(BaseStrategy):
    """Bollinger Band mean reversion with RSI confirmation."""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14,
                 rsi_oversold: float = 30.0, rsi_overbought: float = 70.0):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self._upper = None
        self._middle = None
        self._lower = None
        self._rsi = None

    def init(self, data):
        self.data = data
        self.signals = []
        close = data['close']
        self._upper, self._middle, self._lower = bollinger_bands(close, self.bb_period, self.bb_std)
        self._rsi = rsi(close, self.rsi_period)

    def next(self, bar_idx: int) -> Signal:
        price = self.data['close'][bar_idx]
        ts = self.data['timestamp'][bar_idx]
        
        upper = self._upper[bar_idx]
        lower = self._lower[bar_idx]
        middle = self._middle[bar_idx]
        rsi_val = self._rsi[bar_idx]
        
        if any(v != v for v in [upper, lower, middle, rsi_val]):  # NaN check
            return Signal('HOLD', price, ts, 0.0)
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # BUY: price at/below lower band + RSI oversold
        if price <= lower and rsi_val < self.rsi_oversold:
            signal_type = 'BUY'
            penetration = (lower - price) / lower if lower > 0 else 0.0
            confidence = min(1.0, 0.5 + penetration * 10)
        
        # SELL: price at/above upper band + RSI overbought
        elif price >= upper and rsi_val > self.rsi_overbought:
            signal_type = 'SELL'
            penetration = (price - upper) / upper if upper > 0 else 0.0
            confidence = min(1.0, 0.5 + penetration * 10)
        
        # EXIT signal: price near middle band (mean reversion complete)
        elif abs(price - middle) < (upper - middle) * 0.05:
            signal_type = 'HOLD'
        
        sig = Signal(signal_type, price, ts, confidence,
                     metadata={'upper': upper, 'lower': lower, 'middle': middle, 'rsi': rsi_val})
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {
            'bb_period': self.bb_period, 'bb_std': self.bb_std,
            'rsi_period': self.rsi_period, 'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought
        }
