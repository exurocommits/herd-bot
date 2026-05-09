import numpy as np

class KeltnerVolatilityStrategy:
    """
    Keltner Volatility Strategy.
    Keltner channel width = EMA(20) ± 2*ATR(10).
    Width = 4*ATR.
    Avg width over 50 bars.
    BUY on breakout above upper Keltner when width > 2x average width.
    SELL on breakdown below lower when width > 2x average.
    Confidence: width / avg_width - 1.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.data = data
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.window_ema = 20
        self.window_atr = 10
        self.window_avg = 50
        
        self.signals = []

    def _ema(self, arr, window):
        alpha = 2 / (window + 1)
        ema = np.zeros_like(arr)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
        return ema

    def _atr(self, high, low, close, window):
        tr_list = []
        for i in range(len(close)):
            if i == 0:
                tr_list.append(high[i] - low[i])
                continue
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        tr = np.array(tr_list)
        
        atr = np.zeros_like(tr)
        for i in range(len(tr)):
            if i < window:
                atr[i] = np.mean(tr[:i+1])
            else:
                # Simple moving average for ATR
                atr[i] = np.mean(tr[i-window+1:i+1])
        return atr

    def next(self, bar_idx):
        if bar_idx < max(self.window_ema, self.window_atr, self.window_avg):
            return {"type": "HOLD", "price": float(self.close[bar_idx]), "confidence": 0.0, "metadata": {}}

        # We need EMA and ATR calculated up to current bar
        # For efficiency in a real backtester, these would be precomputed
        ema20 = self._ema(self.close, self.window_ema)
        atr10 = self._atr(self.high, self.low, self.close, self.window_atr)
        
        upper_keltner = ema20[bar_idx] + (2 * atr10[bar_idx])
        lower_keltner = ema20[bar_idx] - (2 * atr10[bar_idx])
        
        # Width = 4 * ATR
        current_width = 4 * atr10[bar_idx]
        
        # Avg width over 50 bars
        # We'd ideally track a rolling avg of the width
        # Here we calculate the historical widths to get the average
        widths = np.array([4 * atr10[i] for i in range(max(0, bar_idx - self.window_avg + 1), bar_idx + 1)])
        avg_width = np.mean(widths)

        signal_type = "HOLD"
        confidence = 0.0

        # BUY: breakout above upper Keltner when width > 2x avg
        if self.close[bar_idx] > upper_keltner and current_width > 2 * avg_width:
            signal_type = "BUY"
            confidence = (current_width / avg_width) - 1
            
        # SELL: breakdown below lower when width > 2x avg
        elif self.close[bar_idx] < lower_keltner and current_width > 2 * avg_width:
            signal_type = "SELL"
            confidence = (current_width / avg_width) - 1

        return {
            "type": signal_type,
            "price": float(self.close[bar_idx]),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "metadata": {"width": float(current_width), "avg_width": float(avg_width)}
        }

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {
            "ema_window": self.window_ema,
            "atr_window": self.window_atr,
            "avg_window": self.window_avg
        }
