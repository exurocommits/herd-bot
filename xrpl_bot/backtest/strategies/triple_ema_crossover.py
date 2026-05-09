import numpy as np

class TripleEMACrossover:
    """
    Triple EMA Crossover strategy.
    Fast(5)/Mid(13)/Slow(34) EMA.
    BUY when fast > mid > slow (bullish alignment).
    SELL when fast < mid < slow.
    Confidence: distance between fastest and slowest EMA / ATR.
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
        
        self.fast_ema = self._calculate_ema(self.close, 5)
        self.mid_ema = self._calculate_ema(self.close, 13)
        self.slow_ema = self._calculate_ema(self.close, 34)
        self.atr = self._calculate_atr(self.high, self.low, self.close, 14)
        
        self.current_idx = 0
        self.signals = []

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
        if bar_idx >= self.n or bar_idx < 34:
            return None
            
        self.current_idx = bar_idx
        price = self.close[bar_idx]
        atr_val = self.atr[bar_idx] if self.atr[bar_idx] > 0 else 1e-9
        
        f = self.fast_ema[bar_idx]
        m = self.mid_ema[bar_idx]
        s = self.slow_ema[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0
        
        # BUY when fast > mid > slow
        if f > m > s:
            signal_type = "BUY"
            confidence = min(abs(f - s) / atr_val, 1.0)
        # SELL when fast < mid < slow
        elif f < m < s:
            signal_type = "SELL"
            confidence = min(abs(f - s) / atr_val, 1.0)
            
        signal = {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {
                "fast_ema": float(f),
                "mid_ema": float(m),
                "slow_ema": float(s)
            }
        }
        self.signals.append(signal)
        return signal

    def get_signals(self):
        """Returns the list of generated signals."""
        return self.signals

    def get_params(self):
        """Returns the strategy parameters."""
        return {"fast": 5, "mid": 13, "slow": 34}
