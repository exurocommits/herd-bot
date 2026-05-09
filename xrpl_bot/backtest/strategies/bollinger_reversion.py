import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy, Signal

class BollingerReversion(BaseStrategy):
    """
    Strategy 4: Bollinger Reversion
    Bollinger Bands (period=20, std_dev=2.0)
    Price touches lower band + RSI < 30 -> BUY
    Price touches upper band + RSI > 70 -> SELL
    Price at middle band -> close position (HOLD)
    Confidence: how far past band edge
    """
    def __init__(self, period=20, std_dev=2.0, rsi_period=14):
        super().__init__()
        self.period = period
        self.std_dev = std_dev
        self.rsi_period = rsi_period
        self.upper_band = None
        self.lower_band = None
        self.middle_band = None
        self.rsi = None

    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(close)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rsis = np.zeros_like(close)
        # Initial RSI
        if down == 0:
            rsis[period] = 100
        else:
            rsis[period] = 100 - (100 / (1 + up / down))
            
        for i in range(period + 1, len(close)):
            delta = deltas[i-1]
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
                
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down == 0:
                rsis[i] = 100
            else:
                rsis[i] = 100 - (100 / (1 + up / down))
        return rsis

    def init(self, data: Dict[str, np.ndarray]):
        super().init(data)
        close = self.data['close']
        
        # Bollinger Bands
        self.middle_band = np.zeros_like(close)
        self.upper_band = np.zeros_like(close)
        self.lower_band = np.zeros_like(close)
        
        for i in range(self.period - 1, len(close)):
            window = close[i - self.period + 1 : i + 1]
            self.middle_band[i] = np.mean(window)
            std = np.std(window)
            self.upper_band[i] = self.middle_band[i] + (self.std_dev * std)
            self.lower_band[i] = self.middle_band[i] - (self.std_dev * std)
            
        self.rsi = self._calculate_rsi(close, self.rsi_period)

    def next(self, bar: Dict[str, Any]) -> Signal:
        idx = self.current_index
        if idx >= len(self.data['close']):
            return Signal('HOLD', bar['close'], bar['timestamp'], 0.0)

        current_price = bar['close']
        upper = self.upper_band[idx]
        lower = self.lower_band[idx]
        middle = self.middle_band[idx]
        rsi_val = self.rsi[idx]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        if current_price <= lower and rsi_val < 30:
            signal_type = 'BUY'
            confidence = min(1.0, (lower - current_price) / (lower if lower != 0 else 1.0) + 0.5) # Heuristic
        elif current_price >= upper and rsi_val > 70:
            signal_type = 'SELL'
            confidence = min(1.0, (current_price - upper) / (upper if upper != 0 else 1.0) + 0.5)
        elif abs(current_price - middle) < (upper - middle) * 0.05: # Near middle band
            signal_type = 'HOLD' # Close position logic
            
        sig = Signal(
            type=signal_type,
            price=current_price,
            timestamp=bar['timestamp'],
            confidence=confidence,
            metadata={'rsi': rsi_val, 'upper': upper, 'lower': lower}
        )
        self.signals.append(sig)
        return sig

    def get_params(self) -> Dict[str, Any]:
        return {'period': self.period, 'std_dev': self.std_dev, 'rsi_period': self.rsi_period}
