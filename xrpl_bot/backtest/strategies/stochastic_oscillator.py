import numpy as np

class StochasticOscillator:
    \"\"\"
    Stochastic Oscillator Strategy.
    %K(14) = (close - low14) / (high14 - low14) * 100.
    %D = 3-period SMA of %K.
    BUY when %K < 20 and crosses above %D.
    SELL when %K > 80 and crosses below %D.
    Confidence: distance from extreme / 100.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        Initialize with data.
        data: dict containing 'close', 'high', 'low' as numpy arrays.
        \"\"\"
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.period = 14
        self.d_period = 3
        self.data_len = len(self.close)
        
        self.k_values = np.zeros(self.data_len)
        self.d_values = np.zeros(self.data_len)
        self.signals = []
        self.current_pos = "HOLD"
        
        self._compute_indicators()

    def _compute_indicators(self):
        for i in range(self.data_len):
            if i >= self.period - 1:
                low_min = np.min(self.low[i - self.period + 1 : i + 1])
                high_max = np.max(self.high[i - self.period + 1 : i + 1])
                diff = high_max - low_min
                if diff != 0:
                    self.k_values[i] = ((self.close[i] - low_min) / diff) * 100
                else:
                    self.k_values[i] = 50.0
            else:
                self.k_values[i] = 50.0

        # Compute %D (SMA of %K)
        for i in range(self.data_len):
            if i >= self.period + self.d_period - 2:
                self.d_values[i] = np.mean(self.k_values[i - self.d_period + 1 : i + 1])
            else:
                self.d_values[i] = 50.0

    def next(self, bar_idx):
        \"\"\"
        Process next bar.
        Returns signal dict.
        \"\"\"
        if bar_idx < self.period + self.d_period:
            return {"type": "HOLD", "price": self.close[bar_idx], "confidence": 0.0, "metadata": {}}

        k = self.k_values[bar_idx]
        d = self.d_values[bar_idx]
        k_prev = self.k_values[bar_idx - 1]
        d_prev = self.d_values[bar_idx - 1]
        price = self.close[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0

        # BUY: %K < 20 and crosses above %D
        if self.current_pos == "HOLD":
            if k < 20 and k_prev <= d_prev and k > d:
                signal_type = "BUY"
                # Confidence: distance from extreme / 100. Extreme is 0 or 100? 
                # Let's say distance from 0 or 100 towards the signal side.
                # If K is 10, it's 10 away from 0.
                confidence = min(k, 100 - k) / 100.0
                self.current_pos = "BUY"
        
        # SELL: %K > 80 and crosses below %D
        elif self.current_pos == "BUY":
            if k > 80 and k_prev >= d_prev and k < d:
                signal_type = "SELL"
                confidence = min(k, 100 - k) / 100.0
                self.current_pos = "HOLD"

        res = {"type": signal_type, "price": float(price), "confidence": float(confidence), "metadata": {"k": float(k), "d": float(d)}}
        if signal_type != "HOLD":
            self.signals.append(res)
        return res

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"period": self.period, "d_period": self.d_period}
