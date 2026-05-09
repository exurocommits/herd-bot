import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy, Signal

class DonchianBreakoutTrend(BaseStrategy):
    """
    Strategy 3: Donchian Breakout Trend
    Track highest high and lowest low over lookback=20
    Price breaks above upper channel -> BUY
    Price breaks below lower channel -> SELL
    Confidence: breakout strength (how far past the channel)
    """
    def __init__(self, lookback=20):
        super().__init__()
        self.lookback = lookback
        self.upper_channel = None
        self.lower_channel = None

    def init(self, data: Dict[str, np.ndarray]):
        super().init(data)
        high = self.data['high']
        low = self.data['low']
        
        self.upper_channel = np.zeros_like(high)
        self.lower_channel = np.zeros_like(low)
        
        for i in range(len(high)):
            start = max(0, i - self.lookback)
            # The channel is calculated from the PREVIOUS bars to avoid look-ahead bias
            if i > 0:
                self.upper_channel[i] = np.max(high[start:i])
                self.lower_channel[i] = np.min(low[start:i])
            else:
                self.upper_channel[i] = high[i]
                self.lower_channel[i] = low[i]

    def next(self, bar: Dict[str, Any]) -> Signal:
        idx = self.current_index
        if idx >= len(self.data['close']):
            return Signal('HOLD', bar['close'], bar['timestamp'], 0.0)

        current_price = bar['close']
        upper = self.upper_channel[idx]
        lower = self.lower_channel[idx]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        if current_price > upper and upper != 0:
            signal_type = 'BUY'
            # Breakout strength: percentage above upper channel
            confidence = min(1.0, (current_price - upper) / upper if upper != 0 else 0.0)
        elif current_price < lower and lower != 0:
            signal_type = 'SELL'
            # Breakout strength: percentage below lower channel
            confidence = min(1.0, (lower - current_price) / lower if lower != 0 else 0.0)

        sig = Signal(
            type=signal_type,
            price=current_price,
            timestamp=bar['timestamp'],
            confidence=confidence,
            metadata={'upper': upper, 'lower': lower}
        )
        self.signals.append(sig)
        return sig

    def get_params(self) -> Dict[str, Any]:
        return {'lookback': self.lookback}
