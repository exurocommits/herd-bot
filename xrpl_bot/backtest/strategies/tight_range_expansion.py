import numpy as np

class TightRangeExpansionStrategy:
    """
    Identify 5-bar ranges where (high - low) < 0.5 * ATR(14). 
    BUY on close > range high. 
    SELL on close < range low. 
    Stop at opposite end of range. 
    Confidence: range tightness (1 - range/ATR).
    """
    def __init__(self, data):
        self.data = data
        self.atr_period = 14
        self.range_period = 5
        self.position = None  # 'LONG', 'SHORT', or None
        self.range_high = 0.0
        self.range_low = 0.0
        self.signals = []
        self.idx = 0
        self.atr = self._compute_atr()

    def _compute_atr(self):
        high = self.data[:, 1]
        low = self.data[:, 2]
        close = self.data[:, 3]
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(close)
        for i in range(self.atr_period, len(close)):
            atr[i] = np.mean(tr[i-self.atr_period:i])
        return atr

    def next(self, bar):
        curr_idx = self.idx
        if curr_idx < max(self.atr_period, self.range_period):
            self.idx += 1
            return {"type": "HOLD", "price": bar[3], "confidence": 0.0, "metadata": {}}

        current_close = bar[3]
        current_high = bar[1]
        current_low = bar[2]
        current_atr = self.atr[curr_idx]
        
        sig = {"type": "HOLD", "price": current_close, "confidence": 0.0, "metadata": {}}

        # Exit logic
        if self.position == 'LONG':
            if current_low < self.range_low:
                self.signals.append({"type": "SELL", "price": current_close, "confidence": 1.0, "metadata": {"reason": "stop_hit"}})
                self.position = None
        elif self.position == 'SHORT':
            if current_high > self.range_high:
                self.signals.append({"type": "BUY", "price": current_close, "confidence": 1.0, "metadata": {"reason": "stop_hit"}})
                self.position = None

        # Entry logic
        if self.position is None:
            # Check for tight range in last 5 bars
            recent_highs = self.data[curr_idx-self.range_period:curr_idx, 1]
            recent_lows = self.data[curr_idx-self.range_period:curr_idx, 2]
            r_high = np.max(recent_highs)
            r_low = np.min(recent_lows)
            r_range = r_high - r_low

            if r_range < 0.5 * current_atr and current_atr > 0:
                # Potential range found
                if current_close > r_high:
                    conf = 1.0 - (r_range / current_atr)
                    self.position = 'LONG'
                    self.range_high = r_high
                    self.range_low = r_low
                    sig = {"type": "BUY", "price": current_close, "confidence": conf, "metadata": {"range": (r_low, r_high)}}
                    self.signals.append(sig)
                elif current_close < r_low:
                    conf = 1.0 - (r_range / current_atr)
                    self.position = 'SHORT'
                    self.range_high = r_high
                    self.range_low = r_low
                    sig = {"type": "SELL", "price": current_close, "confidence": conf, "metadata": {"range": (r_low, r_high)}}
                    self.signals.append(sig)

        self.idx += 1
        return sig

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"atr_period": self.atr_period, "range_period": self.range_period}
