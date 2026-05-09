import numpy as np

class BollingerBandwidthStrategy:
    """
    Bollinger Bandwidth Strategy.
    BB width = (upper - lower) / middle.
    Track 120-bar minimum width.
    BUY when width at 120-day low AND expanding (width > previous bar's width).
    SELL on width contraction after expansion peak.
    Confidence: expansion rate / avg expansion rate.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.data = data
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.window = 20
        self.min_width_period = 120
        
        # Precompute indicators
        self.sma = self._rolling_mean(self.close, self.window)
        self.std = self._rolling_std(self.close, self.window)
        self.upper = self.sma + (2 * self.std)
        self.lower = self.sma - (2 * self.std)
        self.middle = self.sma
        
        # Bandwidth: (upper - lower) / middle
        # Avoid division by zero
        self.width = np.divide(self.upper - self.lower, self.middle, 
                               out=np.zeros_like(self.upper), 
                               where=self.middle != 0)
        
        self.signals = []
        self.expansion_peaks = [] # Track peaks to detect contraction

    def _rolling_mean(self, arr, window):
        weights = np.ones(window) / window
        return np.convolve(arr, weights, mode='valid')

    def _rolling_std(self, arr, window):
        # Simplified rolling std for numpy
        res = np.zeros_like(arr)
        for i in range(window, len(arr)):
            res[i] = np.std(arr[i-window:i])
        return res

    def next(self, bar_idx):
        """
        :param bar_idx: current index
        :return: dict signal
        """
        if bar_idx < max(self.window, self.min_width_period):
            return {"type": "HOLD", "price": float(self.close[bar_idx]), "confidence": 0.0, "metadata": {}}

        current_width = self.width[bar_idx]
        prev_width = self.width[bar_idx-1]
        
        # Check 120-bar minimum width
        min_width_val = np.min(self.width[bar_idx-self.min_width_period:bar_idx])
        
        # Expansion rate calculation (for confidence)
        # Using a moving average of expansion rates (diffs)
        expansion_rates = np.diff(self.width)
        avg_expansion = np.mean(expansion_rates[max(0, bar_idx-120):bar_idx]) if bar_idx > 120 else 0.01

        signal_type = "HOLD"
        confidence = 0.0
        metadata = {}

        # BUY logic: width at 120-day low AND expanding
        if current_width <= min_width_val and current_width > prev_width:
            signal_type = "BUY"
            rate = current_width - prev_width
            confidence = min(1.0, rate / avg_expansion if avg_expansion > 0 else 0.0)
        
        # SELL logic: contraction after expansion peak
        # Simple peak detection: if prev_width was > prev_prev_width and current < prev
        elif bar_idx > 2 and prev_width > self.width[bar_idx-2] and current_width < prev_width:
            # We assume it was expanding. To be more robust we'd track peaks.
            signal_type = "SELL"
            confidence = 0.5 # Default for contraction

        return {
            "type": signal_type,
            "price": float(self.close[bar_idx]),
            "confidence": float(confidence),
            "metadata": {"width": float(current_width), "min_width": float(min_width_val)}
        }

    def get_signals(self):
        # This is a placeholder if the user wants to run the whole sequence
        return self.signals

    def get_params(self):
        return {"window": self.window, "min_width_period": self.min_width_period}
