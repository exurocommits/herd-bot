import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy, Signal

class RSIDivergencePlay(BaseStrategy):
    """
    Strategy 5: RSI Divergence Play
    RSI period=14, overbought=70, oversold=30
    Bullish divergence: price lower low + RSI higher low -> BUY
    Bearish divergence: price higher high + RSI lower high -> SELL
    Detect divergence over last 20 bars
    Confidence: magnitude of divergence
    """
    def __init__(self, rsi_period=14, lookback=20):
        super().__init__()
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.rsi = None

    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(close)
        rsi = np.zeros_like(close)
        
        # Using a simpler RSI calculation for inline
        for i in range(period, len(close)):
            window = close[i-period:i]
            up = 0
            down = 0
            for j in range(1, len(window)):
                diff = window[j] - window[j-1]
                if diff > 0:
                    up += diff
                else:
                    down -= diff
            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100 - (100 / (1 + rs))
        return rsi

    def init(self, data: Dict[str, np.ndarray]):
        super().init(data)
        self.rsi = self._calculate_rsi(self.data['close'], self.rsi_period)

    def next(self, bar: Dict[str, Any]) -> Signal:
        idx = self.current_index
        if idx < self.lookback + 1 or idx >= len(self.data['close']):
            return Signal('HOLD', bar['close'], bar['timestamp'], 0.0)

        current_price = bar['close']
        current_rsi = self.rsi[idx]
        
        # Divergence detection window
        window_idx = range(idx - self.lookback, idx)
        prices = self.data['close'][window_idx]
        rsis = self.rsi[window_idx]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # Bullish Divergence: Price Lower Low + RSI Higher Low
        # Find local min in price and RSI
        price_min_idx = np.argmin(prices)
        rsi_min_idx = np.argmin(rsis)
        
        # A simple check: if the current RSI is higher than the previous local minimum RSI, 
        # and the current price is lower than the previous local minimum price.
        # This is a very simplified divergence detector for the task.
        
        # Look for a previous trough
        found_bullish = False
        for i in range(len(prices) - 2):
            # check if prices[i] is a local min
            if prices[i] < prices[i+1] and prices[i] < prices[i-1 if i>0 else 0]:
                # check if current is a local min and higher RSI
                if current_price < prices[i] and current_rsi > rsis[i]:
                    found_bullish = True
                    break
        
        if found_bullish:
            signal_type = 'BUY'
            confidence = min(1.0, abs(current_rsi - rsis[np.argmin(rsis)]) / 100.0)

        # Bearish Divergence: Price Higher High + RSI Lower High
        found_bearish = False
        for i in range(len(prices) - 2):
            # check if prices[i] is a local max
            if prices[i] > prices[i+1] and prices[i] > prices[i-1 if i>0 else 0]:
                # check if current is a local max and lower RSI
                if current_price > prices[i] and current_rsi < rsis[i]:
                    found_bearish = True
                    break
        
        if found_bearish:
            signal_type = 'SELL'
            confidence = min(1.0, abs(rsis[np.argmax(rsis)] - current_rsi) / 100.0)

        sig = Signal(
            type=signal_type,
            price=current_price,
            timestamp=bar['timestamp'],
            confidence=confidence,
            metadata={'rsi': current_rsi}
        )
        self.signals.append(sig)
        return sig

    def get_params(self) -> Dict[str, Any]:
        return {'rsi_period': self.rsi_period, 'lookback': self.lookback}
