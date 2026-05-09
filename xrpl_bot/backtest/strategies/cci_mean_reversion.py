import numpy as np

class CCIMeanReversion:
    \"\"\"
    CCI Mean Reversion Strategy.
    CCI(20) = (TP - SMA_TP) / (0.015 * mean_dev).
    BUY when CCI < -100 and crosses back above.
    SELL when CCI > +100 and crosses below.
    Exit at 0 cross.
    Confidence: |CCI| / 200.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        Initialize with data.
        data: dict containing 'close', 'high', 'low' as numpy arrays.
        \"\"\"
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.period = 20
        self.data_len = len(self.close)
        
        self.cci_values = np.zeros(self.data_len)
        self.signals = []
        self.current_pos = "HOLD"
        
        self._compute_indicators()

    def _compute_indicators(self):
        tp = (self.high + self.low + self.close) / 3
        for i in range(self.data_len):
            if i >= self.period - 1:
                window_tp = tp[i - self.period + 1 : i + 1]
                sma_tp = np.mean(window_tp)
                mean_dev = np.mean(np.abs(window_tp - sma_tp))
                if mean_dev != 0:
                    self.cci_values[i] = (tp[i] - sma_tp) / (0.015 * mean_dev)
                else:
                    self.cci_values[i] = 0.0
            else:
                self.cci_values[i] = 0.0

    def next(self, bar_idx):
        \"\"\"
        Process next bar.
        Returns signal dict.
        \"\"\"
        if bar_idx < self.period:
            return {"type": "HOLD", "price": self.close[bar_idx], "confidence": 0.0, "metadata": {}}

        cci = self.cci_values[bar_idx]
        cci_prev = self.cci_values[bar_idx - 1]
        price = self.close[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0

        # BUY: CCI < -100 and crosses back above
        if self.current_pos == "HOLD":
            if cci_prev < -100 and cci > -100:
                signal_type = "BUY"
                confidence = min(abs(cci), 200.0) / 200.0
                self.current_pos = "BUY"
        
        # SELL: CCI > +100 and crosses below
        elif self.current_pos == "BUY":
            if cci_prev > 100 and cci < 100:
                signal_type = "SELL"
                confidence = min(abs(cci), 200.0) / 200.0
                self.current_pos = "HOLD"
            # Exit at 0 cross
            elif (cci_prev < 0 and cci > 0) or (cci_prev > 0 and cci < 0):
                # The prompt says "Exit at 0 cross". 
                # If we are in BUY (long) and CCI crosses 0.
                # Let's check if we are coming from below or above? 
                # Usually mean reversion exits when returning to the mean (0).
                # Let's assume exit if it crosses 0.
                signal_type = "SELL"
                confidence = min(abs(cci), 200.0) / 200.0
                self.current_pos = "HOLD"

        res = {"type": signal_type, "price": float(price), "confidence": float(confidence), "metadata": {"cci": float(cci)}}
        if signal_type != "HOLD":
            self.signals.append(res)
        return res

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"period": self.period}
