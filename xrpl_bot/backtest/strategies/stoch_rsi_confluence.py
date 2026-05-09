"""
Stochastic + RSI Confluence Strategy — Pure Python, no numpy.
Combines Stochastic oscillator and RSI for high-confidence reversal signals.
BUY: Stochastic %K < 20 + RSI < 35 + %K crosses above %D
SELL: Stochastic %K > 80 + RSI > 65 + %K crosses below %D
"""
import math
from .base_strategy import BaseStrategy, Signal
from .indicators import stochastic, rsi, cross_above, cross_below, sma


class StochRSIConfluence(BaseStrategy):
    """Stochastic + RSI confluence reversal strategy."""
    
    def __init__(self, stoch_k: int = 14, stoch_d: int = 3, rsi_period: int = 14,
                 oversold: float = 20.0, overbought: float = 80.0,
                 rsi_low: float = 35.0, rsi_high: float = 65.0):
        super().__init__()
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self._k_line = None
        self._d_line = None
        self._rsi = None

    def init(self, data):
        self.data = data
        self.signals = []
        close = data['close']
        self._k_line, self._d_line = stochastic(
            data['high'], data['low'], close, self.stoch_k, self.stoch_d
        )
        self._rsi = rsi(close, self.rsi_period)

    def next(self, bar_idx: int) -> Signal:
        price = self.data['close'][bar_idx]
        ts = self.data['timestamp'][bar_idx]
        
        k = self._k_line[bar_idx]
        d = self._d_line[bar_idx]
        rsi_val = self._rsi[bar_idx]
        
        if any(v != v for v in [k, d, rsi_val]):
            return Signal('HOLD', price, ts, 0.0)
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # BUY: Oversold stochastic + RSI confirmation + crossover
        if (k < self.oversold and rsi_val < self.rsi_low and
                cross_above(self._k_line, self._d_line, bar_idx)):
            signal_type = 'BUY'
            # Confidence: how oversold (lower = more confident)
            stoch_conf = (self.oversold - k) / self.oversold
            rsi_conf = (self.rsi_low - rsi_val) / self.rsi_low
            confidence = min(1.0, (stoch_conf + rsi_conf) / 2 + 0.3)
        
        # SELL: Overbought stochastic + RSI confirmation + crossunder
        elif (k > self.overbought and rsi_val > self.rsi_high and
              cross_below(self._k_line, self._d_line, bar_idx)):
            signal_type = 'SELL'
            stoch_conf = (k - self.overbought) / (100 - self.overbought)
            rsi_conf = (rsi_val - self.rsi_high) / (100 - self.rsi_high)
            confidence = min(1.0, (stoch_conf + rsi_conf) / 2 + 0.3)
        
        sig = Signal(signal_type, price, ts, confidence,
                     metadata={'stoch_k': k, 'stoch_d': d, 'rsi': rsi_val})
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {
            'stoch_k': self.stoch_k, 'stoch_d': self.stoch_d,
            'rsi_period': self.rsi_period, 'oversold': self.oversold,
            'overbought': self.overbought
        }
