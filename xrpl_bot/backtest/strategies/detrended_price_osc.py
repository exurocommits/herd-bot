import numpy as np

class DetrendedPriceOscillator:
    \"\"\"
    Detrended Price Oscillator (DPO) Strategy.
    DPO(20) = close - SMA(close, period/2+1)[shifted period/2].
    Zero-line oscillator.
    BUY when DPO crosses above 0 from below.
    SELL when DPO crosses below 0 from above.
    Confidence: |DPO| / std(DPO).
    \"\"\"
    def __init__(self, data):
        \"\"\"
        Initialize with data.
        data: dict containing 'close' as numpy array.
        \"\"\"
        self.close = data['close']
        self.period = 20
        self.data_len = len(self.close)
        
        self.dpo_values = np.zeros(self.data_len)
        self.dpo_std = 0.0
        self.signals = []
        self.current_pos = "HOLD"
        
        self._compute_indicators()

    def _compute_indicators(self):
        # DPO(20) = close - SMA(close, period/2+1)[shifted period/2]
        # period/2 + 1 = 20/2 + 1 = 11
        # shift = 20/2 = 10
        sma_period = self.period // 2 + 1
        shift = self.period // 2
        
        if self.data_len > sma_period + shift:
            # Calculate SMA of close with sma_period
            sma_values = np.zeros(self.data_len)
            for i in range(sma_period - 1, self.data_len):
                sma_values[i] = np.mean(self.close[i - sma_period + 1 : i + 1])
            
            # DPO = close - SMA[shifted]
            # The SMA value used for index i is the SMA(close, sma_period) from index (i - shift)
            for i in range(shift, self.data_len):
                self.dpo_values[i] = self.close[i] - sma_values[i - shift]
            
            # Standard deviation of DPO for confidence
            # Only use non-zero DPO values for std calculation
            valid_dpo = self.dpo_values[shift:self.data_len]
            if len(valid_dpo) > 0:
                self.dpo_std = np.std(valid_dpo)
        else:
            self.dpo_std = 1.0 # Avoid division by zero

    def next(self, bar_idx):
        \"\"\"
        Process next bar.
        Returns signal dict.
        \"\"\"
        if bar_idx < self.period: # Sufficient buffer
            return {"type": "HOLD", "price": self.close[bar_idx], "confidence": 0.0, "metadata": {}}

        dpo = self.dpo_values[bar_idx]
        dpo_prev = self.dpo_values[bar_idx - 1]
        price = self.close[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0

        # BUY when DPO crosses above 0 from below
        if self.current_pos == "HOLD":
            if dpo_prev < 0 and dpo > 0:
                signal_type = "BUY"
                confidence = abs(dpo) / self.dpo_std if self.dpo_std != 0 else 0.0
                self.current_pos = "BUY"
        
        # SELL when DPO crosses below 0 from above
        elif self.current_pos == "BUY":
            if dpo_prev > 0 and dpo < 0:
                signal_type = "SELL"
                confidence = abs(dpo) / self.dpo_std if self.dpo_std != 0 else 0.0
                self.current_pos = "HOLD"

        res = {"type": signal_type, "price": float(price), "confidence": float(confidence), "metadata": {"dpo": float(dpo)}}
        if signal_type != "HOLD":
            self.signals.append(res)
        return res

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"period": self.period}
