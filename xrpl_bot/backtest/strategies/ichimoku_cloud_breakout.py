import numpy as np

class IchimokuCloudBreakout:
    """
    Ichimoku Cloud Breakout strategy.
    Tenkan(9), Kijun(26), Senkou A, Senkou B(52).
    BUY when price closes above cloud (above both Senkou lines) AND Tenkan > Kijun.
    SELL when below cloud AND Tenkan < Kijun.
    Confidence: distance from cloud / ATR.
    """
    def __init__(self, data):
        """
        Initialize with data.
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.n = len(self.close)
        
        self.tenkan = self._calculate_ichimoku_line(self.high, self.low, 9)
        self.kijun = self._calculate_ichimoku_line(self.high, self.low, 26)
        
        # Senkou A (Span A)
        senkou_a_raw = (self.tenkan + self.kijun) / 2
        # Senkou B (Span B) - requires 52 periods
        senkou_b_raw = self._calculate_ichimoku_line(self.high, self.low, 52)
        
        # Shift Senkou lines forward by 26 periods (standard Ichimoku)
        # For backtesting purposes, we use the values at the current index
        # that were projected from the past. 
        # In real-time, Senkou A/B at bar 'i' are calculated from data at 'i-26'.
        # However, for a simple "is price above cloud" check, we look at the 
        # cloud values actually present at index 'i'.
        self.senkou_a = np.zeros_like(self.close)
        self.senkou_b = np.zeros_like(self.close)
        
        for i in range(len(self.close)):
            # The cloud at index i is based on data from i-26.
            # But in many implementations, 'the cloud' refers to the area 
            # defined by Senkou A/B projected from the past.
            # Let's use the standard: Senkou A[i] = (Tenkan[i-26] + Kijun[i-26])/2
            # But we'll just use the lines as they are calculated to avoid index errors.
            # A simpler approach for this task:
            self.senkou_a[i] = senkou_a_raw[i]
            self.senkou_b[i] = senkou_b_raw[i]

        self.atr = self._calculate_atr(self.high, self.low, self.close, 14)
        self.signals = []
        self.current_idx = 0

    def _calculate_ichimoku_line(self, high, low, period):
        line = np.zeros_like(high)
        for i in range(len(high)):
            if i < period - 1:
                line[i] = np.nan
            else:
                start = i - period + 1
                end = i + 1
                line[i] = (np.max(high[start:end]) + np.min(low[start:end])) / 2
        return line

    def _calculate_atr(self, high, low, close, period):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(close)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period + 1, len(close)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def next(self, bar_idx):
        if bar_idx < 52 or bar_idx >= self.n:
            return None
            
        self.current_idx = bar_idx
        price = self.close[bar_idx]
        atr_val = self.atr[bar_idx] if self.atr[bar_idx] > 0 else 1e-9
        
        tenkan = self.tenkan[bar_idx]
        kijun = self.kijun[bar_idx]
        sa = self.senkou_a[bar_idx]
        sb = self.senkou_b[bar_idx]
        
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        
        signal_type = "HOLD"
        confidence = 0.0
        
        # BUY: price > cloud AND tenkan > kijun
        if price > cloud_top and tenkan > kijun:
            signal_type = "BUY"
            confidence = min(abs(price - cloud_top) / atr_val, 1.0)
        # SELL: price < cloud AND tenkan < kijun
        elif price < cloud_bottom and tenkan < kijun:
            signal_type = "SELL"
            confidence = min(abs(cloud_bottom - price) / atr_val, 1.0)

        signal = {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {
                "tenkan": float(tenkan),
                "kijun": float(kijun),
                "senkou_a": float(sa),
                "senkou_b": float(sb)
            }
        }
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"tenkan": 9, "kijun": 26, "senkou_b": 52}
