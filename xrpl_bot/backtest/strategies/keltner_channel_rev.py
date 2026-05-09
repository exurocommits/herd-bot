import numpy as np

class KeltnerChannelRev:
    \"\"\"
    Keltner Channel Reversal Strategy.
    midline = EMA(20)
    upper = midline + 1.5 * ATR(10)
    lower = midline - 1.5 * ATR(10)
    BUY at lower band touch.
    SELL at upper band.
    Exit at EMA midline.
    Confidence: distance from EMA / channel width.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        Initialize with data.
        data: dict containing 'close', 'high', 'low' as numpy arrays.
        \"\"\"
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.ema_period = 20
        self.atr_period = 10
        self.data_len = len(self.close)
        
        self.midline = np.zeros(self.data_len)
        self.upper_band = np.zeros(self.data_len)
        self.lower_band = np.zeros(self.data_len)
        self.signals = []
        self.current_pos = "HOLD"
        
        self._compute_indicators()

    def _compute_indicators(self):
        # EMA(20)
        alpha = 2 / (self.ema_period + 1)
        ema = np.zeros(self.data_len)
        if self.data_len > 0:
            ema[0] = self.close[0]
            for i in range(1, self.data_len):
                ema[i] = (self.close[i] * alpha) + (ema[i-1] * (1 - alpha))
        self.midline = ema

        # ATR(10)
        # TR = max(H-L, abs(H-Cp), abs(L-Cp))
        tr = np.zeros(self.data_len)
        for i in range(1, self.data_len):
            tr[i] = max(self.high[i] - self.low[i], 
                        abs(self.high[i] - self.close[i-1]), 
                        abs(self.low[i] - self.close[i-1]))
        
        atr = np.zeros(self.data_len)
        if self.data_len >= self.atr_period:
            # Simple SMA for ATR as per common implementation if not specified
            for i in range(self.atr_period - 1, self.data_len):
                atr[i] = np.mean(tr[i - self.atr_period + 1 : i + 1])
        
        for i in range(self.data_len):
            self.upper_band[i] = self.midline[i] + 1.5 * atr[i]
            self.lower_band[i] = self.midline[i] - 1.5 * atr[i]

    def next(self, bar_idx):
        \"\"\"
        Process next bar.
        Returns signal dict.
        \"\"\"
        if bar_idx < max(self.ema_period, self.atr_period):
            return {"type": "HOLD", "price": self.close[bar_idx], "confidence": 0.0, "metadata": {}}

        price = self.close[bar_idx]
        mid = self.midline[bar_idx]
        upper = self.upper_band[bar_idx]
        lower = self.lower_band[bar_idx]
        channel_width = upper - lower
        
        signal_type = "HOLD"
        confidence = 0.0

        # BUY at lower band touch
        if self.current_pos == "HOLD":
            if price <= lower:
                signal_type = "BUY"
                # Confidence: distance from EMA / channel width
                confidence = abs(price - mid) / channel_width if channel_width != 0 else 0.0
                self.current_pos = "BUY"
        
        # SELL at upper band
        elif self.current_pos == "BUY":
            if price >= upper:
                signal_type = "SELL"
                confidence = abs(price - mid) / channel_width if channel_width != 0 else 0.0
                self.current_pos = "HOLD"
            # Exit at EMA midline
            elif (price <= mid <= self.close[bar_idx-1]) or (price >= mid >= self.close[bar_idx-1]):
                # Crossing or touching the midline
                signal_type = "SELL"
                confidence = abs(price - mid) / channel_width if channel_width != 0 else 0.0
                self.current_pos = "HOLD"

        res = {"type": signal_type, "price": float(price), "confidence": float(confidence), "metadata": {"mid": float(mid), "upper": float(upper), "lower": float(lower)}}
        if signal_type != "HOLD":
            self.signals.append(res)
        return res

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"ema_period": self.ema_period, "atr_period": self.atr_period}
