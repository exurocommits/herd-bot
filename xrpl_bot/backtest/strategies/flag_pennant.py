import numpy as np

class FlagPennantStrategy:
    """
    Pole: 5-bar move > 2*ATR in one direction. 
    Flag: next 3-5 bars consolidate in opposite direction with declining volume. 
    BUY on breakout above flag high. 
    SELL on breakout below flag low. 
    Confidence: pole size / flag size ratio.
    """
    def __init__(self, data):
        self.data = data
        self.atr_period = 14
        self.position = None
        self.signals = []
        self.idx = 0
        self.atr = self._compute_atr()
        self.state = "SEARCHING" # SEARCHING, FLAG
        self.pole_size = 0.0
        self.flag_high = 0.0
        self.flag_low = 0.0
        self.flag_start_idx = 0

    def _compute_atr(self):
        high, low, close = self.data[:, 1], self.data[:, 2], self.data[:, 3]
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr = np.zeros_like(close)
        for i in range(self.atr_period, len(close)):
            atr[i] = np.mean(tr[i-self.atr_period:i])
        return atr

    def next(self, bar):
        idx = self.idx
        if idx < self.atr_period + 6:
            self.idx += 1
            return {"type": "HOLD", "price": bar[3], "confidence": 0.0, "metadata": {}}

        c_close, c_high, c_low, c_vol, c_atr = bar[3], bar[1], bar[2], bar[4], self.atr[idx]
        sig = {"type": "HOLD", "price": c_close, "confidence": 0.0, "metadata": {}}

        # Exit logic
        if self.position == 'LONG':
            if c_low < self.flag_low:
                self.signals.append({"type": "SELL", "price": c_close, "confidence": 1.0, "metadata": {"reason": "flag_break"}})
                self.position = None
                self.state = "SEARCHING"
            else:
                self.idx += 1
                return sig
        elif self.position == 'SHORT':
            if c_high > self.flag_high:
                self.signals.append({"type": "BUY", "price": c_close, "confidence": 1.0, "metadata": {"reason": "flag_break"}})
                self.position = None
                self.state = "SEARCHING"
            else:
                self.idx += 1
                return sig

        # Pattern logic
        if self.state == "SEARCHING":
            pole_data = self.data[idx-5:idx]
            pole_move = pole_data[-1, 3] - pole_data[0, 3]
            if abs(pole_move) > 2 * c_atr:
                self.state = "FLAG"
                self.pole_size = abs(pole_move)
                self.flag_start_idx = idx
                self.flag_high = c_high
                self.flag_low = c_low
                
        elif self.state == "FLAG":
            duration = idx - self.flag_start_idx
            if duration > 5:
                self.state = "SEARCHING"
            else:
                self.flag_high = max(self.flag_high, c_high)
                self.flag_low = min(self.flag_low, c_low)
                
                if c_close > self.flag_high:
                    conf = self.pole_size / (self.flag_high - self.flag_low) if (self.flag_high - self.flag_low) > 0 else 1.0
                    self.position = 'LONG'
                    self.signals.append({"type": "BUY", "price": c_close, "confidence": conf, "metadata": {}})
                    self.state = "SEARCHING"
                elif c_close < self.flag_low:
                    conf = self.pole_size / (self.flag_high - self.flag_low) if (self.flag_high - self.flag_low) > 0 else 1.0
                    self.position = 'SHORT'
                    self.signals.append({"type": "SELL", "price": c_close, "confidence": conf, "metadata": {}})
                    self.state = "SEARCHING"

        self.idx += 1
        return sig

    def get_signals(self): return self.signals
    def get_params(self): return {"atr_period": self.atr_period}
