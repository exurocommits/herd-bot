import numpy as np

class HistoricalVolRatioStrategy:
    """
    Historical Volatility Ratio Strategy.
    Short vol = std(returns, 10). Long vol = std(returns, 100). Ratio = short/long.
    Ratio < 0.5 = vol compression.
    BUY on price breakout above 20-bar high when ratio < 0.5 (vol compression breakout).
    SELL when ratio > 1.5 (vol expansion exhaustion).
    Confidence: |1 - ratio|.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close' as numpy array.
        """
        self.data = data
        self.close = data['close']
        self.returns = np.diff(np.log(self.close), prepend=np.log(self.close[0]))
        
        self.short_window = 10
        self.long_window = 100
        self.breakout_window = 20
        
        self.signals = []

    def _rolling_std(self, arr, window):
        res = np.zeros_like(arr)
        for i in range(window, len(arr)):
            res[i] = np.std(arr[i-window:i])
        return res

    def next(self, bar_idx):
        if bar_idx < max(self.long_window, self.breakout_window):
            return {"type": "HOLD", "price": float(self.close[bar_idx]), "confidence": 0.0, "metadata": {}}

        # Calculate volatilities at the current bar
        # returns[:bar_idx+1] ensures we look at history up to current bar
        hist_returns = self.returns[:bar_idx+1]
        
        # Note: for true real-time, we should use a rolling window approach.
        # Here we approximate the std of the window ending at bar_idx
        short_vol = np.std(hist_returns[max(0, bar_idx - self.short_window):bar_idx+1])
        long_vol = np.std(hist_returns[max(0, bar_idx - self.long_window):bar_idx+1])
        
        ratio = short_vol / long_vol if long_vol != 0 else 1.0
        
        # Breakout high: max of previous 20 bars (excluding current)
        breakout_high = np.max(self.close[bar_idx - self.breakout_window : bar_idx])

        signal_type = "HOLD"
        confidence = 0.0
        
        # BUY: breakout above 20-bar high AND ratio < 0.5
        if self.close[bar_idx] > breakout_high and ratio < 0.5:
            signal_type = "BUY"
            confidence = abs(1.0 - ratio)
            
        # SELL: ratio > 1.5
        elif ratio > 1.5:
            signal_type = "SELL"
            confidence = abs(1.0 - ratio)

        return {
            "type": signal_type,
            "price": float(self.close[bar_idx]),
            "confidence": float(min(1.0, confidence)),
            "metadata": {"ratio": float(ratio), "short_vol": float(short_vol), "long_vol": float(long_vol)}
        }

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "breakout_window": self.breakout_window
        }
