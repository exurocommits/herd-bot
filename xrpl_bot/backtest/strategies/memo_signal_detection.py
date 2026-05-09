import numpy as np

class MemoSignalDetection:
    """
    Simulate memo-based signals: random "whale" events detected 
    when volume > 5x average AND price moves > 1.5*ATR.
    
    BUY on whale accumulation signal (volume spike + price up).
    SELL on whale distribution (volume spike + price down).
    Confidence: volume_ratio / 5.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'high', 'low', 'volume'
        """
        self.closes = np.array(data['close'])
        self.highs = np.array(data['high'])
        self.lows = np.array(data['low'])
        self.volumes = np.array(data['volume'])
        self.lookback = 20

    def _calculate_atr(self, period=14):
        if len(self.closes) < period:
            return 0.0
        # Simple ATR simulation
        highs = self.highs[-period:]
        lows = self.lows[-period:]
        closes_prev = self.closes[-period-1:-1]
        
        tr = np.maximum(highs - lows, 
                        np.maximum(np.abs(highs - closes_prev), 
                                   np.abs(lows - closes_prev)))
        return np.mean(tr)

    def next(self, bar):
        price_close = bar['close']
        price_high = bar['high']
        price_low = bar['low']
        volume = bar['volume']

        self.closes = np.append(self.closes, price_close)
        self.highs = np.append(self.highs, price_high)
        self.lows = np.append(self.lows, price_low)
        self.volumes = np.append(self.volumes, volume)

        if len(self.volumes) < self.lookback:
            return {"type": "HOLD", "price": float(price_close), "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        avg_volume = np.mean(self.volumes[-self.lookback:-1])
        atr = self._calculate_atr()
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        price_move = abs(price_close - self.closes[-2]) if len(self.closes) > 1 else 0

        signal_type = "HOLD"
        # Condition: volume > 5x average AND price moves > 1.5*ATR
        if volume_ratio > 5 and price_move > 1.5 * atr:
            if price_close > self.closes[-2]:
                signal_type = "BUY"  # Accumulation
            else:
                signal_type = "SELL" # Distribution

        confidence = min(volume_ratio / 5.0, 1.0)

        return {
            "type": signal_type,
            "price": float(price_close),
            "confidence": float(confidence),
            "metadata": {
                "volume_ratio": float(volume_ratio),
                "atr": float(atr)
            }
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"lookback": self.lookback}
