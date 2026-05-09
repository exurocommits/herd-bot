import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy, Signal

class EMACrossGolden(BaseStrategy):
    """
    Strategy 1: EMA Cross Golden
    Fast EMA (9) crosses above slow EMA (21) -> BUY
    Fast EMA crosses below slow EMA -> SELL
    Confidence: based on distance between EMAs relative to ATR
    """
    def __init__(self, fast_period=9, slow_period=21, atr_period=14):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        
        self.fast_ema = None
        self.slow_ema = None
        self.atr = None
        self.prev_fast = None
        self.prev_slow = None

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def init(self, data: Dict[str, np.ndarray]):
        super().init(data)
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        self.fast_ema = self._calculate_ema(close, self.fast_period)
        self.slow_ema = self._calculate_ema(close, self.slow_period)
        self.atr = self._calculate_atr(high, low, close, self.atr_period)
        
        self.prev_fast = self.fast_ema[0]
        self.prev_slow = self.slow_ema[0]

    def next(self, bar: Dict[str, Any]) -> Signal:
        idx = self.current_index
        if idx >= len(self.data['close']):
            return Signal('HOLD', bar['close'], bar['timestamp'], 0.0)

        current_fast = self.fast_ema[idx]
        current_slow = self.slow_ema[idx]
        current_atr = self.atr[idx]
        current_price = bar['close']
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # Check crossover
        if self.prev_fast <= self.prev_slow and current_fast > current_slow:
            signal_type = 'BUY'
            # Confidence: distance between EMAs / ATR
            diff = current_fast - current_slow
            confidence = min(1.0, diff / (current_atr if current_atr > 0 else 1.0))
        elif self.prev_fast >= self.prev_slow and current_fast < current_slow:
            signal_type = 'SELL'
            diff = current_slow - current_fast
            confidence = min(1.0, diff / (current_atr if current_atr > 0 else 1.0))

        self.prev_fast = current_fast
        self.prev_slow = current_slow
        
        sig = Signal(
            type=signal_type,
            price=current_price,
            timestamp=bar['timestamp'],
            confidence=confidence,
            metadata={'fast_ema': current_fast, 'slow_ema': current_slow, 'atr': current_atr}
        )
        self.signals.append(sig)
        return sig

    def get_params(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'atr_period': self.atr_period
        }
