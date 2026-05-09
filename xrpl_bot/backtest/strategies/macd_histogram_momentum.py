import numpy as np

class MACDHistogramMomentum:
    """
    MACD Histogram Momentum strategy.
    MACD(12,26,9).
    BUY on histogram flip from negative to positive AND increasing for 3 consecutive bars.
    SELL on flip from positive to negative AND decreasing for 3 bars.
    Confidence: histogram magnitude / ATR.
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
        
        # MACD Components
        ema12 = self._calculate_ema(self.close, 12)
        ema26 = self._calculate_ema(self.close, 26)
        self.macd_line = ema12 - ema26
        self.signal_line = self._calculate_ema(self.macd_line, 9)
        self.histogram = self.macd_line - self.signal_line
        
        self.atr = self._calculate_atr(self.high, self.low, self.close, 14)
        self.signals = []
        self.current_idx = 0

    def _calculate_ema(self, series, period):
        ema = np.zeros_like(series)
        alpha = 2 / (period + 1)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

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
        """
        Processes the next bar.
        :param bar_idx: Index of the current bar.
        :return: dict signal or None.
        """
        if bar_idx < 30 or bar_idx >= self.n:
            return None
            
        self.current_idx = bar_idx
        price = self.close[bar_idx]
        hist = self.histogram[bar_idx]
        atr_val = self.atr[bar_idx] if self.atr[bar_idx] > 0 else 1e-9
        
        signal_type = "HOLD"
        confidence = 0.0
        
        # Check for 3 consecutive increasing bars in histogram (for BUY)
        # and 3 consecutive decreasing bars (for SELL)
        # Requirement: Flip from neg to pos (BUY) or pos to neg (SELL)
        
        # BUY condition: Flip from negative to positive AND increasing for 3 consecutive bars
        # Hist[i] > 0, Hist[i-1] <= 0, Hist[i] > Hist[i-1], Hist[i-1] > Hist[i-2], Hist[i-2] > Hist[i-3]
        # (Wait, the prompt says 'increasing for 3 consecutive bars', implying 3 bars of upward momentum)
        # Let's interpret as: Hist[i] > Hist[i-1] > Hist[i-2] > Hist[i-3] AND flip happened.
        
        is_increasing = (self.histogram[bar_idx] > self.histogram[bar_idx-1] > self.histogram[bar_idx-2] > self.histogram[bar_idx-3])
        is_decreasing = (self.histogram[bar_idx] < self.histogram[bar_idx-1] < self.histogram[bar_idx-2] < self.histogram[bar_idx-3])
        
        # Flip check
        flipped_to_pos = (self.histogram[bar_idx] > 0 and self.histogram[bar_idx-1] <= 0)
        flipped_to_neg = (self.histogram[bar_idx] < 0 and self.histogram[bar_idx-1] >= 0)

        if flipped_to_pos and is_increasing:
            signal_type = "BUY"
            confidence = min(abs(hist) / atr_val, 1.0)
        elif flipped_to_neg and is_decreasing:
            signal_type = "SELL"
            confidence = min(abs(hist) / atr_val, 1.0)

        signal = {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {
                "macd": float(self.macd_line[bar_idx]),
                "signal": float(self.signal_line[bar_idx]),
                "histogram": float(hist)
            }
        }
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"fast": 12, "slow": 26, "signal": 9}
