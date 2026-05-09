import numpy as np

class WilliamsPercentR:
    \"\"\"
    Williams %R Strategy.
    %R(14) = (high14 - close)/(high14 - low14) * -100.
    BUY when %R crosses above -80 (coming from below).
    SELL when crosses below -20.
    Exit at -50.
    Confidence: |%R - -50| / 50.
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
        self.data_len = len(self.close)
        
        self.r_values = np.zeros(self.data_len)
        self.signals = []
        self.current_pos = "HOLD"
        
        self._compute_indicators()

    def _compute_indicators(self):
        for i in range(self.data_len):
            if i >= self.period - 1:
                high_max = np.max(self.high[i - self.period + 1 : i + 1])
                low_min = np.min(self.low[i - self.period + 1 : i + 1])
                diff = high_max - low_min
                if diff != 0:
                    self.r_values[i] = ((high_max - self.close[i]) / diff) * -100
                else:
                    self.r_values[i] = -50.0
            else:
                self.r_values[i] = -50.0

    def next(self, bar_idx):
        \"\"\"
        Process next bar.
        Returns signal dict.
        \"\"\"
        if bar_idx < self.period:
            return {"type": "HOLD", "price": self.close[bar_idx], "confidence": 0.0, "metadata": {}}

        r = self.r_values[bar_idx]
        r_prev = self.r_values[bar_idx - 1]
        price = self.close[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0

        # BUY: %R crosses above -80 (coming from below)
        if self.current_pos == "HOLD":
            if r_prev < -80 and r > -80:
                signal_type = "BUY"
                confidence = abs(r - (-50)) / 50.0
                self.current_pos = "BUY"
        
        # SELL: crosses below -20
        elif self.current_pos == "BUY":
            if r < -20 and r_prev >= -20:
                signal_type = "SELL"
                confidence = abs(r - (-50)) / 50.0
                self.current_pos = "HOLD"
            # Exit at -50
            elif r > -50 and r_prev <= -50: # Assuming crossover logic or just hitting it
                 # The prompt says "Exit at -50". If we are in BUY and R crosses -50...
                 # Let's refine: if we are BUYing and R crosses -50, we exit.
                 # But wait, crossing -50 from below or above? Usually, mean reversion implies returning to center.
                 # Let's assume: if we hit -50 while in a position.
                 pass # handled below

        # Re-evaluating logic for EXIT at -50
        if self.current_pos == "BUY":
             # The prompt says: BUY at -80 cross up. SELL at -20 cross down. Exit at -50.
             # This implies an exit condition if it hits -50.
             if abs(r - (-50)) < 0.1: # Close to -50
                 # Note: This might conflict with the 'SELL' if SELL is also a close.
                 # Let's implement it as a specific exit type if needed, or just a SELL.
                 # Let's use SELL for simplicity of the signal dict.
                 signal_type = "SELL"
                 confidence = abs(r - (-50)) / 50.0
                 self.current_pos = "HOLD"

        res = {"type": signal_type, "price": float(price), "confidence": float(confidence), "metadata": {"wpr": float(r)}}
        if signal_type != "HOLD":
            self.signals.append(res)
        return res

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"period": self.period}
